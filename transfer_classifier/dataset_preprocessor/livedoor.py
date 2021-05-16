from pathlib import Path

import numpy as np
import pandas as pd
from datasets import DownloadManager, load_dataset
from datasets.arrow_dataset import Dataset
from transfer_classifier.dataset_preprocessor.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)
from transformers.tokenization_utils import PreTrainedTokenizer


class Livedoor(ClassificationDatasetPreprocessor):
    NUM_CLASS = 9

    def __init__(
        self,
        input_column: str,
        label_column: str,
        tokenizer: PreTrainedTokenizer = None,
        truncation: bool = True,
        max_length: int = 512,
        padding: str = "max_length",
        batched: bool = True,
        lang: str = "ja",
        validation_size: float = 0.2,
        test_size: float = 0.2,
    ):
        super().__init__(
            input_column=input_column,
            label_column=label_column,
            tokenizer=tokenizer,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            label_function=None,
            batched=batched,
            lang=lang,
        )
        self.validation_size = validation_size
        self.test_size = test_size

    def save(self, force: bool = False) -> Path:
        url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
        manager = DownloadManager()
        expanded_path = manager.download_and_extract(url)
        text_path = Path(expanded_path).joinpath("text")
        dataset_path = Path(expanded_path).joinpath("dataset.csv")

        if dataset_path.exists() and not force:
            return dataset_path.parent

        dataset = []
        for label, directory in enumerate([d for d in text_path.iterdir() if d.is_dir()]):
            for text in directory.glob("*.txt"):
                if text.name.startswith(directory.name):
                    lines = []
                    with text.open(mode="r", encoding="utf-8") as r:
                        lines = r.readlines()

                    url, time, title, *bodies = lines
                    body = "\n".join(bodies)
                    sample = {"labels": label, "label_name": directory.name}
                    sample["url"] = url
                    sample["time"] = time
                    sample["title"] = title
                    sample["body"] = body
                    dataset.append(sample)

        df = pd.DataFrame(dataset)
        df.to_csv(dataset_path, index=False)

        validation_size = int(len(df) * self.validation_size)
        test_size = int(len(df) * self.test_size)

        if len(df) <= (validation_size + test_size):
            raise Exception("Amount of train dataset is under 0.")

        validation, test, train = np.split(
            df.sample(frac=1, random_state=42),
            [
                validation_size,
                (validation_size + test_size),
            ],
        )

        dataset_root = dataset_path.parent
        train.to_csv(dataset_root.joinpath("train.csv"), index=False)
        validation.to_csv(dataset_root.joinpath("validation.csv"), index=False)
        test.to_csv(dataset_root.joinpath("test.csv"), index=False)
        return dataset_root

    def load(
        self,
        split: str = "train",
    ) -> Dataset:

        dataset_root = self.save()
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(dataset_root.joinpath("train.csv")),
                "validation": str(dataset_root.joinpath("validation.csv")),
                "test": str(dataset_root.joinpath("test.csv")),
            },
        )
        return dataset[split]
