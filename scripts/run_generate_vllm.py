import sys
import subprocess
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import scripts.batch_convert_lit_checkpoint as ckpts

chosen = ckpts.oasst1_dolly_lingua

for entry in chosen:
    subprocess.run(['python',
                    'generate/vllm_full.py',
                    # '--model_name',
                    entry[0],
                    # '--checkpoint_name',
                    entry[2],
                    # '--dataset_name',
                    entry[3]])
