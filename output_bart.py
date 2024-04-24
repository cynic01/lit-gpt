import pandas as pd
import numpy as np
from glob import glob

def main(csv_glob: str):
    master_df = pd.DataFrame()

    for csv_filename in sorted(glob(csv_glob)):
        df = pd.read_csv(csv_filename)
        master_df = pd.concat([master_df, df.BARTScore.dropna().describe().rename(csv_filename)], axis=1)

    print (master_df.T)

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)