import sys
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Literal, Optional

import pandas as pd
import torch
from vllm import LLM, SamplingParams

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

def main(
    model_name: str,
    checkpoint_name: str,
    dataset_name: str,
    max_new_tokens: int = 512,
    top_p: float = 1,
    top_k: int = -1,
    temperature: float = 0.7,
    devices: int = 1,
) -> None:
    converted_model_path = f"out/converted/{model_name}-{dataset_name}-{checkpoint_name}"
    checkpoint_dir = f"checkpoints/EleutherAI/{model_name}"
    dataset_path = f"data/{dataset_name}/test.pt"
    out_path = f"out/inference/{model_name}-{dataset_name}-{checkpoint_name}.pkl"

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_new_tokens)

    print(f"Loading model {converted_model_path!r} with {checkpoint_dir}")
    t0 = time.perf_counter()
    llm = LLM(model=converted_model_path,
              tokenizer=checkpoint_dir,
              tensor_parallel_size=devices,
              swap_space=0)
    print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    
    tok = llm.get_tokenizer()
    tok.add_tokens(["<|system|>", "<|user|>", "<|assistant|>"], special_tokens=True)

    test_data = torch.load(dataset_path)
    print("Length of test data", len(test_data))

    y = llm.generate(prompt_token_ids=[entry['input_ids_no_response'].tolist() for entry in test_data],
                     sampling_params=sampling_params)

    input = tok.batch_decode([entry['input_ids_no_response'] for entry in test_data], skip_special_tokens=True)
    output = [entry.outputs[0].text for entry in y]
    label = tok.batch_decode([entry['labels'][len(entry['input_ids_no_response']):] for entry in test_data], skip_special_tokens=True)

    df = pd.DataFrame({'input': input, 'output': output, 'label': label}, dtype='string')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
