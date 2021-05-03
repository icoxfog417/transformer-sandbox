from typing import Any, Dict, List

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer


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
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        target: str = "review_title",
        batched: bool = True,
    ) -> Dataset:
        def encode(examples: Dict[str, List[Any]]) -> BatchEncoding:
            tokenized = tokenizer(
                examples[target],
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
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        target: str = "review_title",
        batched: bool = True,
    ) -> Dataset:
        tokenized = self.tokenize(dataset, tokenizer, target, batched)
        labeled = self.labels(tokenized, batched)
        filtered = labeled.filter(lambda example: example["labels"] >= 0)
        columns = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in filtered.column_names:
            columns += ["token_type_ids"]

        filtered.set_format(
            type="torch",
            columns=columns,
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
