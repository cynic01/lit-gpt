import pandas as pd
import numpy as np
from glob import glob

def main(json_glob: str):
    master_df = pd.DataFrame()

    for json_filename in sorted(glob(json_glob)):
        data = pd.read_json(json_filename)
        stats = data.score.apply(pd.Series).stack().str.split('/').apply(pd.Series)[0].str.strip().replace({'n':'None'}).apply(eval).unstack().mean()
        master_df = pd.concat([master_df, stats.rename(json_filename)], axis=1)

    print (master_df.T)

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)