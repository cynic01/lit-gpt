import pandas as pd
import subprocess
import json
import os

THEMIS_TEMP_DIR = 'out/themis_temp'
THEMIS_OUT_DIR = "out/themis_out"

def get_themis_template(aspect, input, output, seg_id):
    return {
    "task": "Dialogue Response Generation",
    "aspect": aspect,
    "source_des": "Dialogue Context",
    "source": input,
    "target_des": "Response",
    "target": output,   
    "seg_id": seg_id
}

THEMIS_ASPECTS = [
    "Context Maintenance: Does the response serve as a valid continuation of the dialogue context (conversation history)?", 
    "Interestingness: Is the response dull or interesting?",
    "Knowledge Use: Given the fact that the response is conditioned on, how well does the response use that fact?",
    "Naturalness: Does the response seem to be something that a person would naturally say?"
]

def main(pickle: str):
  df = pd.read_pickle(pickle)

  query_filename = pickle.replace('out/inference', THEMIS_TEMP_DIR).replace('.pkl', '.json')
  if os.path.exists('evaluation_' + query_filename.replace(THEMIS_TEMP_DIR, THEMIS_OUT_DIR)):
    print ('output for ', query_filename, 'already exists, exiting')
    return

  subprocess.run(['mkdir', THEMIS_TEMP_DIR])
    
  queries = []
  for i, row in df.iterrows():
    for aspect in THEMIS_ASPECTS:
        queries.append(get_themis_template(aspect, row.input + '\n', row.output + '\n', i))
    
  with open(query_filename, 'w') as f:
      json.dump(queries, f)
    
  subprocess.run(['python', 'Themis/eval.py', 
                  '--model', "PKU-ONELab/Themis",
                  '--test_dir', THEMIS_TEMP_DIR,
                  '--output_dir', THEMIS_OUT_DIR,
                  '--tensor_parallel_size', '1'])

  subprocess.run(['rm', '-r', THEMIS_TEMP_DIR])

if __name__ == '__main__':
    from jsonargparse.cli import CLI
    CLI(main)