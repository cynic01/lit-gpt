import sys
import subprocess
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

for model_name in [f'pythia-{nparams}-deduped' for nparams in ('70m', '160m', '410m', '1b', '1.4b', '2.8b')]:
    for dataset_name in ['sharegpt', 'oasst1-dolly', 'oasst1']:
        subprocess.run(['python',
                        'generate/vllm_base.py',
                        # '--model_name',
                        model_name,
                        # '--dataset_name',
                        dataset_name])
