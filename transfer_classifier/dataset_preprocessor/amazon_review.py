from typing import Dict

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transfer_classifier.dataset_preprocessor.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)
from transformers.tokenization_utils import PreTrainedTokenizer


class AmazonReview(ClassificationDatasetPreprocessor):
    NUM_CLASS = 2

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
    ):
        super().__init__(
            input_column=input_column,
            label_column=label_column,
            tokenizer=tokenizer,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            label_function=self.to_label,
            batched=batched,
            lang=lang,
        )

    def load(self, split: str, filter_medium_star: bool = True, shuffle: bool = True) -> Dataset:
        dataset = load_dataset("amazon_reviews_multi", self.lang, split=split)
        if filter_medium_star:
            dataset = dataset.filter(lambda example: example["stars"] in (1, 5))
        if shuffle:
            return dataset.shuffle()
        else:
            return dataset

    def to_label(self, star: int) -> int:
        if 1 < star < 5:
            return -1
        elif star == 1:
            return 0
        else:
            return 1

    def statistics(self, formatted: Dataset) -> Dict[str, int]:
        positives = len([e for e in formatted if e["labels"].item() == 1])
        negatives = len([e for e in formatted if e["labels"].item() == 0])

        return {
            "total": len(formatted),
            "positive": positives,
            "negative": negatives,
        }
