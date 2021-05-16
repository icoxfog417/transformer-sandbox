import shutil
import os
from typing import Dict, Any
from pathlib import Path
import pytest
from datasets import load_dataset
from transfer_classifier.augmentor.augmented_dataset import AugmentedDataset
from transformers import AutoTokenizer


@pytest.fixture
def test_path():
    path = Path(os.path.dirname(__file__)).joinpath("test_augmented")
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)
    shutil.rmtree(str(path))


class TestAugmentedDataset:
    def test_save_dataset(self, test_path) -> None:
        dataset = AugmentedDataset(test_path, "test_save")
        original = load_dataset("amazon_reviews_multi", "ja", split="test[:10]")
        augmented = load_dataset("amazon_reviews_multi", "ja", split="test[:10]")
        validation = load_dataset("amazon_reviews_multi", "ja", split="test[:11]")
        test = load_dataset("amazon_reviews_multi", "ja", split="test[:12]")

        def augment(example: Dict[str, Any]) -> Dict[str, Any]:
            example["review_title"] = "XXXX" + example["review_title"]
            return example

        augmented = augmented.map(augment)
        print(augmented[0]["review_title"])
        dataset.save_dataset(original, augmented, validation, test)
        loaded_dataset = dataset.load_dataset()
        assert len(loaded_dataset["train"]) == 20
        assert (
            len(loaded_dataset["train"].filter(lambda e: e["_kind"] == "augmented"))
            == 10
        )
        assert len(loaded_dataset["validation"]) == 11
        assert len(loaded_dataset["test"]) == 12
        assert loaded_dataset["train"][10]["review_title"].startswith("X")
