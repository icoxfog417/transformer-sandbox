from typing import Any, Dict, List
from typing import Callable
from typing import Optional

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer


class ClassificationDatasetPreprocessor:
    def __init__(
        self,
        input_column: str,
        label_column: str,
        tokenizer: PreTrainedTokenizer,
        truncation: bool = True,
        max_length: int = 512,
        padding: str = "max_length",
        label_function: Optional[Callable[[Any], int]] = None,
        batched: bool = True,
        lang: str = "ja",
    ) -> None:
        self.input_column = input_column
        self.label_column = label_column
        self.tokenizer = tokenizer
        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.label_function = label_function
        self.batched = batched
        self.lang = lang

    def tokenize(
        self,
        dataset: Dataset,
    ) -> Dataset:
        def encode(examples: Dict[str, List[Any]]) -> BatchEncoding:
            tokenized = self.tokenizer(
                examples[self.input_column],
                truncation=self.truncation,
                max_length=self.max_length,
                padding=self.padding,
            )
            return tokenized

        return dataset.map(encode, batched=self.batched)

    def format_labels(
        self,
        dataset: Dataset,
    ) -> Dataset:

        if self.label_function is not None:
            _label_function: Callable[[Any], int] = self.label_function
        else:
            _label_function = lambda example: int(example[self.label_column])

        def encode(examples: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
            labels = {
                "labels": np.array(
                    [_label_function(s) for s in examples[self.label_column]]
                )
            }
            return labels

        return dataset.map(encode, batched=self.batched)

    def format(self, dataset: Dataset) -> Dataset:
        tokenized = self.tokenize(dataset)
        label_formatted = self.format_labels(tokenized)
        columns = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in label_formatted.column_names:
            columns += ["token_type_ids"]

        label_formatted.set_format(
            type="torch",
            columns=columns,
        )
        return label_formatted
