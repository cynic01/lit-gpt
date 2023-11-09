# Format: model_name, run_name, checkpoint_name, dataset_name

# all sharegpt checkpoints are on Vox server
sharegpt = [
    ("pythia-2.8b-deduped", "abominable-broomstick-372", "iter-000549-ckpt", "sharegpt"),
    ("pythia-2.8b-deduped", "abominable-broomstick-372", "iter-000049-ckpt", "sharegpt"),
    ("pythia-1.4b-deduped", "headless-specter-368", "iter-000299-ckpt", "sharegpt"),
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
    ("pythia-70m-deduped", "frightful-specter-346", "iter-000349-ckpt", "sharegpt")
]

# oasst1 + dolly checkpoints are on Lingua for >= 1b models
oasst1_dolly_lingua = [
    ("pythia-2.8b-deduped", "soft-cosmos-295", "iter-000099-ckpt", "oasst1-dolly"),
    ("pythia-2.8b-deduped", "soft-cosmos-295", "iter-000399-ckpt", "oasst1-dolly"),
    ("pythia-1.4b-deduped", "decent-snowball-309", "iter-002799-ckpt", "oasst1-dolly"),
    ("pythia-1.4b-deduped", "decent-snowball-309", "iter-000099-ckpt", "oasst1-dolly"),
    ("pythia-1b-deduped", "icy-sky-311", "iter-000499-ckpt", "oasst1-dolly"),
    ("pythia-1b-deduped", "icy-sky-311", "iter-000099-ckpt", "oasst1-dolly"),
]

# oasst1 + dolly checkpoints are on Vox for < 1b models
oasst1_dolly_vox = [
    ("pythia-410m-deduped", "denim-blaze-306", "iter-001549-ckpt", "oasst1-dolly"),
    ("pythia-410m-deduped", "denim-blaze-306", "iter-000399-ckpt", "oasst1-dolly"),
    ("pythia-410m-deduped", "denim-blaze-306", "iter-000049-ckpt", "oasst1-dolly"),
    ("pythia-160m-deduped", "lyric-puddle-303", "iter-000549-ckpt", "oasst1-dolly"),
    ("pythia-160m-deduped", "lyric-puddle-303", "iter-000099-ckpt", "oasst1-dolly"),
    ("pythia-70m-deduped", "distinctive-shadow-300", "iter-001299-ckpt", "oasst1-dolly"),
    ("pythia-70m-deduped", "distinctive-shadow-300", "iter-000149-ckpt", "oasst1-dolly"),
]

chosen = sharegpt

if __name__ == '__main__':
    import requests
    from convert_lit_checkpoint import *

    for run in chosen:
        model_name, run_name, checkpoint_name, dataset_name = run
        output_path = Path(f"out/converted/{model_name}-{dataset_name}-{checkpoint_name}/pytorch_model.bin")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_lit_checkpoint(checkpoint_path=Path(f"out/full/{model_name}-{dataset_name}/{run_name}/{checkpoint_name}.pth"),
                            output_path=output_path,
                            config_path=Path(f"checkpoints/EleutherAI/{model_name}/lit_config.json"))
        response = requests.get(f'https://huggingface.co/EleutherAI/{model_name}/resolve/main/config.json')
        with open(output_path.parent / 'config.json', 'w') as f:
            f.write(response.text)
