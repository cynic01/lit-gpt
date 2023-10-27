import sys
import time
import csv
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Literal, Optional

import lightning as L
import torch
# from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    gptq_quantization,
    load_checkpoint,
)
from scripts.prepare_alpaca import generate_prompt
from finetune.full import get_longest_seq_length

model_name = "pythia-2.8b-deduped"
run_name = "driven-wood-164"
checkpoint_name = "iter-000299-ckpt"
dataset_name = "oasst1"

def main(
    # prompt: str = "What food do lamas eat?",
    # input: str = "",
    finetuned_path: Path = Path(f"out/full/{model_name}-{dataset_name}/{run_name}/{checkpoint_name}.pth"),
    checkpoint_dir: Path = Path(f"checkpoints/EleutherAI/{model_name}"),
    dataset_path: Path = Path(f"data/{dataset_name}/test.pt"),
    out_path: Path = Path(f"out/inference/{model_name}-{dataset_name}-{checkpoint_name}.csv"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT model.
    See `finetune/full.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        finetuned_path: Path to the checkpoint with trained weights, which are the output of
            `finetune/full.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if devices > 1:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantization flag."
            )
        if quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)

    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize is not None and devices > 1:
        raise NotImplementedError
    checkpoint_path = finetuned_path

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    test_data = torch.load(dataset_path)
    print("Length of test data", len(test_data))

    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(test_data)
    # sample = {"instruction": prompt, "input": input}
    # prompt = generate_prompt(sample)
    # encoded = tokenizer.encode(f'<|user|>{prompt}<|assistant|>', device=fabric.device)
    # prompt_length = encoded.size(0)
    max_returned_tokens = longest_seq_length + max_new_tokens

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        csv_writer = csv.writer(f)
        for x, label in tqdm(get_data(fabric, test_data, longest_seq_length), total=len(test_data)):
            t0 = time.perf_counter()
            y = generate(model, x, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
            t = time.perf_counter() - t0

            input = tokenizer.decode(x, skip_special_tokens=True).strip()
            output = tokenizer.decode(y[len(x):], skip_special_tokens=True).strip()#.rstrip('<|endoftext|>')  # model specific
            label = tokenizer.decode(label[len(x):], skip_special_tokens=True).strip()
            # fabric.print(output)
            csv_writer.writerow((input, output, label))

    # tokens_generated = y.size(0) - prompt_length
    # fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


def get_data(
    fabric: L.Fabric, data: List[Dict], longest_seq_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    for entry in data:
        x = entry["input_ids_no_response"].type(torch.int64)
        y = entry["labels"].type(torch.int64)
        if fabric.device.type == "cuda" and x.device.type == "cpu":
            x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
        else:
            x, y = fabric.to_device((x, y))
        yield x, y


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
