import sys
from pathlib import Path

import random
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

def combine(
    destination_path: Path = Path("data/oasst1_dolly"),
    first_dataset: Path = Path("data/oasst1"),
    second_dataset: Path = Path("data/dolly")
):
    """Combine first_dataset with second_dataset and store it in destination_path."""
    destination_path.mkdir(parents=True, exist_ok=True)

    print("Processing train split ...")

    first_train = torch.load(first_dataset / "train.pt")
    print("Length of 1st train data", len(first_train))

    second_train = torch.load(second_dataset / "train.pt")
    print("Length of 2nd train data", len(second_train))

    first_train.extend(second_train)
    random.shuffle(first_train)
    print("Total length of train data", len(first_train))
    torch.save(first_train, destination_path / "train.pt")

    print("Processing test split ...")
    
    first_test = torch.load(first_dataset / "test.pt")
    print("Length of 1st test data", len(first_test))

    second_test = torch.load(second_dataset / "test.pt")
    print("Length of 2nd test data", len(second_test))

    first_test.extend(second_test)
    random.shuffle(first_test)
    print("Total length of test data", len(first_test))
    torch.save(first_test, destination_path / "test.pt")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(combine)
