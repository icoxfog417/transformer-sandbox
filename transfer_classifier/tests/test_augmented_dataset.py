import os
import shutil
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
from datasets import load_dataset
from transfer_classifier.augmented_dataset import AugmentedDataset
from transfer_classifier.dataset_preprocessor.amazon_review import AmazonReview


@pytest.fixture
def test_path() -> Generator[str, None, None]:
    path = Path(os.path.dirname(__file__)).joinpath("test_augmented")
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)
    shutil.rmtree(str(path))


class TestAugmentedDataset:
    def test_save_dataset(self, test_path: str) -> None:
        dataset = AugmentedDataset(test_path, "test_save")
        original = load_dataset("amazon_reviews_multi", "ja", split="test[:10]")
        augmented = load_dataset("amazon_reviews_multi", "ja", split="test[:10]")
        validation = load_dataset("amazon_reviews_multi", "ja", split="test[:11]")
        test = load_dataset("amazon_reviews_multi", "ja", split="test[:12]")

        def augment(example: Dict[str, Any]) -> Dict[str, Any]:
            example["review_title"] = "XXXX" + example["review_title"]
            return example

        augmented = augmented.map(augment)
        review = AmazonReview(input_column="review_title", label_column="stars")
        dataset.save_dataset(review, original, augmented, validation, test)
        loaded_dataset = dataset.load_dataset()
        assert len(loaded_dataset["train"]) == 20
        assert (
            len(
                loaded_dataset["train"].filter(
                    lambda e: e[dataset._kind_column] == "augmented"
                )
            )
            == 10
        )
        assert len(loaded_dataset["validation"]) == 11
        assert len(loaded_dataset["test"]) == 12
        assert loaded_dataset["train"][10][dataset._input_column].startswith("X")
