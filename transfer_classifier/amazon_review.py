import numpy as np
from typing import Dict
from typing import List
from typing import Any
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding


class AmazonReview:
    def __init__(self, lang: str = "ja"):
        self.lang = lang

    def load(self, split: str, shuffle: bool = True) -> Dataset:
        dataset = load_dataset("amazon_reviews_multi", self.lang, split=split)
        if shuffle:
            return dataset.shuffle()
        else:
            return dataset

    def tokenize(
        self, dataset: Dataset, tokenizer: PreTrainedTokenizer, batched: bool = True
    ) -> Dataset:
        def encode(examples: Dict[str, List[Any]]) -> BatchEncoding:
            tokenized = tokenizer(
                examples["review_title"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            return tokenized

        return dataset.map(encode, batched=batched)

    def labels(self, dataset: Dataset, batched: bool = True) -> Dataset:
        def convert_star(star: int) -> int:
            if 1 < star < 5:
                return -1
            elif star == 1:
                return 0
            else:
                return 1

        def encode(examples: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
            labels = {"labels": np.array([convert_star(s) for s in examples["stars"]])}
            return labels

        return dataset.map(encode, batched=batched)

    def format(
        self, dataset: Dataset, tokenizer: PreTrainedTokenizer, batched: bool = True
    ) -> Dataset:
        tokenized = self.tokenize(dataset, tokenizer, batched)
        labeled = self.labels(tokenized, batched)
        filtered = labeled.filter(lambda example: example["labels"] >= 0)
        filtered.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        )
        return filtered

    def statistics(self, formatted: Dataset) -> Dict[str, int]:
        positives = len([e for e in formatted if e["labels"].item() == 1])
        negatives = len([e for e in formatted if e["labels"].item() == 0])

        return {
            "total": len(formatted),
            "positive": positives,
            "negative": negatives,
        }
