from convert_lit_checkpoint import *

# Format: model_name, run_name, checkpoint_name, dataset_name
sharegpt = [("pythia-2.8b-deduped", "abominable-broomstick-372", "iter-000549-ckpt", "sharegpt"),
            ("pythia-2.8b-deduped", "abominable-broomstick-372", "iter-000049-ckpt", "sharegpt"),
            ("pythia-1.4b-deduped", "headless-specter-368", "iter-001749-ckpt", "sharegpt"),
            ("pythia-1.4b-deduped", "headless-specter-368", "iter-000049-ckpt", "sharegpt"),
            ("pythia-1.4b-deduped", "headless-specter-368", "iter-000099-ckpt", "sharegpt"),
            ("pythia-1b-deduped", "creepy-bones-360", "iter-001799-ckpt", "sharegpt"),
            ("pythia-1b-deduped", "creepy-bones-360", "iter-000049-ckpt", "sharegpt"),
            ("pythia-1b-deduped", "creepy-bones-360", "iter-000099-ckpt", "sharegpt"),
            ("pythia-410m-deduped", "ritualistic-banshee-351", "iter-000749-ckpt", "sharegpt"),
            ("pythia-410m-deduped", "ritualistic-banshee-351", "iter-000099-ckpt", "sharegpt"),
            ("pythia-160m-deduped", "mystical-newt-350", "lit_model_finetuned", "sharegpt"),
            ("pythia-160m-deduped", "mystical-newt-350", "iter-000299-ckpt", "sharegpt"),
            ("pythia-160m-deduped", "mystical-newt-350", "iter-000099-ckpt", "sharegpt"),
            ("pythia-70m-deduped", "frightful-specter-346", "lit_model_finetuned", "sharegpt"),
            ("pythia-70m-deduped", "frightful-specter-346", "iter-000349-ckpt", "sharegpt"),]

chosen = sharegpt

for run in chosen:
    model_name, run_name, checkpoint_name, dataset_name = run
    convert_lit_checkpoint(checkpoint_path=Path(f"out/full/{model_name}-{dataset_name}/{run_name}/{checkpoint_name}.pth"),
                           output_path=Path(f"out/converted/{model_name}-{dataset_name}-{checkpoint_name}/pytorch_model.bin"),
                           config_path=Path(f"checkpoints/EleutherAI/{model_name}/lit_config.json"))