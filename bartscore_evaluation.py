import sys; sys.path.append('BartScore')
from pathlib import Path
import pandas as pd
from BARTScore.bart_score import BARTScorer

def main(pickle: Path, device='cpu', batch_size=1, checkpoint='tareknaous/bart-daily-dialog', max_length=2048):
  df = pd.read_pickle(pickle)
  scorer = BARTScorer(device=device, max_length=max_length, checkpoint=checkpoint)
  scores = scorer.score(df.input.tolist(), df.output.tolist(), batch_size=batch_size)
  df['BARTScore'] = scores
  df.to_csv(pickle.with_suffix('.csv'), index=None)


if __name__ == '__main__':
    from jsonargparse.cli import CLI
    CLI(main)