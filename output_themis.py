import pandas as pd
import numpy as np
from glob import glob
import json

THEMIS_ASPECTS = [
    "Context Maintenance: Does the response serve as a valid continuation of the dialogue context (conversation history)?", 
    "Interestingness: Is the response dull or interesting?",
    "Knowledge Use: Given the fact that the response is conditioned on, how well does the response use that fact?",
    "Naturalness: Does the response seem to be something that a person would naturally say?"
]

def main(json_glob: str):
    master_df = pd.DataFrame()
    
    for json_filename in sorted(glob(json_glob)):
        print (json_filename)
        data = json.load(open(json_filename))
        rows = [{'id': i // 4, 
                 'metric': THEMIS_ASPECTS[i % 4].split(':')[0],
                 'analysis': row['Evaluation Outputs'][0]['Analysis'],
                 'rating': row['Evaluation Outputs'][0]['Rating']
                 } for i, row in enumerate(data['Evaluation'])]
        
        df = pd.DataFrame(rows)
        master_df = pd.concat([master_df, df.groupby('metric').rating.mean().rename(json_filename.split('/')[-1])], axis=1)
        
    print(master_df.T)
        
if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)