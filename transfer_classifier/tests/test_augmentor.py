import os

import torch
from datasets import load_dataset
from pytest_mock import MockFixture
from transfer_classifier.augmentor.autoencoder_augmentor import AutoEncoderAugmentor
from transfer_classifier.dataset_preprocessor.amazon_review import AmazonReview
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class TestAugmentor:
    def test_augment(self) -> None:
        review = AmazonReview(input_column="review_title", label_column="stars")
        samples = review.load("validation").select(range(10))

        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        model = AutoModelForMaskedLM.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        review.tokenizer = tokenizer

        augmentor = AutoEncoderAugmentor(model=model, tokenizer=tokenizer)
        augmented = augmentor.augment(samples, review)
        assert len(augmented) > 0
        assert augmentor.__AUGMENTATION_VALID__ not in augmented.features

    def test_augment_with_discriminator(self, mocker: MockFixture) -> None:
        path = os.path.join(os.path.dirname(__file__), "test.csv")
        review = AmazonReview(input_column="review_title", label_column="stars")
        samples = load_dataset("csv", data_files={"train": path})["train"]

        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        model = AutoModelForMaskedLM.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        review.tokenizer = tokenizer

        discriminator = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

        # original: positive, negative, positive, negative, positive
        # augmented: positive, negative, negative, negative, positive
        # matched = 4

        class OutputMock:
            def __init__(self) -> None:
                self.num_call = 0

            def output(self, *args: str, **kwargs: int):  # type: ignore
                if self.num_call == 0:
                    self.num_call += 1
                    return torch.tensor(
                        [
                            [
                                [0.9, -0.9],
                                [-0.9, 0.9],
                                [0.9, -0.9],
                                [-0.9, 0.9],
                                [0.9, -0.9],
                            ]
                        ]
                    )

                else:
                    return torch.tensor(
                        [
                            [
                                [0.9, -0.9],
                                [-0.9, 0.9],
                                [-0.9, 0.9],
                                [-0.9, 0.9],
                                [0.9, -0.9],
                            ]
                        ]
                    )

        output_mock = OutputMock()
        mocker.patch.object(
            type(discriminator),
            "__call__",
            side_effect=output_mock.output,
        )

        augmentor = AutoEncoderAugmentor(model=model, tokenizer=tokenizer)
        augmented = augmentor.augment(samples, review, discriminator=discriminator)
        assert len(augmented) == 4
        assert augmentor.__AUGMENTATION_VALID__ not in augmented.features
