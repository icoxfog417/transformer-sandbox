from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transfer_classifier.dataset_preprocessor.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer


class AugmentedDataset:
    def __init__(self, path: str, directory: str) -> None:
        self._path = path
        self._directory = directory
        self._input_column = "inputs"
        self._label_column = "labels"
        self._kind_column = "_kinds"

    @property
    def path(self) -> Path:
        return Path(self._path).joinpath(self._directory)

    def save_dataset(
        self,
        preprocessor: ClassificationDatasetPreprocessor,
        original: Dataset,
        augmented: Dataset,
        validation: Dataset,
        test: Dataset,
    ) -> None:
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        for split_kind in ["train", "validation", "test"]:
            if split_kind == "train":
                if len(augmented) > 0:
                    dataset = {"original": original, "augmented": augmented}
                else:
                    dataset = {"original": original}
            elif split_kind == "validation":
                dataset = {"original": validation}
            elif split_kind == "test":
                dataset = {"original": test}

            samples = []
            for kind in dataset:
                _dataset = preprocessor.format_labels(dataset[kind])
                for sample in _dataset:
                    _sample = {
                        self._input_column: sample[preprocessor.input_column],
                        self._label_column: sample["labels"],
                        self._kind_column: kind,
                    }
                    samples.append(_sample)

            file_name = f"{split_kind}.csv"
            # Todo: Add logging
            df = pd.DataFrame(samples)
            df.to_csv(self.path.joinpath(file_name), index=False)

    def load_dataset(self) -> Dataset:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.path.joinpath("train.csv")),
                "validation": str(self.path.joinpath("validation.csv")),
                "test": str(self.path.joinpath("test.csv")),
            },
        )
        return dataset

    def format(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        truncation: bool = True,
        max_length: int = 512,
        padding: str = "max_length",
    ) -> Dataset:
        def encode(examples: Dict[str, List[Any]]) -> BatchEncoding:
            tokenized = tokenizer(
                examples[self._input_column],
                truncation=truncation,
                max_length=max_length,
                padding=padding,
            )
            return tokenized

        tokenized = dataset.map(encode)
        columns = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in tokenized.column_names:
            columns += ["token_type_ids"]

        tokenized.set_format(
            type="torch",
            columns=columns,
        )
        return tokenized
