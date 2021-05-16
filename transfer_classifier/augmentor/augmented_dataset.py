from pathlib import Path
from re import split
import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets import load_dataset


class AugmentedDataset:
    def __init__(self, path: str, directory: str) -> None:
        self._path = path
        self._directory = directory

    @property
    def path(self):
        return Path(self._path).joinpath(self._directory)

    def save_dataset(
        self, original: Dataset, augmented: Dataset, validation: Dataset, test: Dataset
    ):
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        columns = original.column_names
        for split_kind in ["train", "validation", "test"]:
            if split_kind == "train":
                dataset = {"original": original, "augmented": augmented}
            elif split_kind == "validation":
                dataset = {"original": validation}
            elif split_kind == "test":
                dataset = {"original": test}

            samples = []
            for kind in dataset:
                for sample in dataset[kind]:
                    _sample = {}
                    for column in columns:
                        _sample[column] = sample[column]

                    _sample["_kind"] = kind
                    samples.append(_sample)

            file_name = f"{split_kind}.csv"
            # Todo: Add logging
            df = pd.DataFrame(samples)
            df.to_csv(self.path.joinpath(file_name), index=False)

    def load_dataset(self):
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.path.joinpath("train.csv")),
                "validation": str(self.path.joinpath("validation.csv")),
                "test": str(self.path.joinpath("test.csv")),
            },
        )
        return dataset
