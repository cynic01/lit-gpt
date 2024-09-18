from fed import fed
import pandas as pd
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

def main(pickle: Path):
  df = pd.read_pickle(pickle)
  model, tokenizer = fed.load_models("microsoft/DialoGPT-large")
  df['fed_scores'] = df.progress_apply(lambda row: fed.evaluate('<|endoftext|> ' + row.input + ' <|endoftext|> ' + row.output, model, tokenizer), axis=1)

  df.to_csv(pickle.with_suffix('.fed.csv'), index=None)

if __name__ == '__main__':
    from jsonargparse.cli import CLI
    CLI(main)