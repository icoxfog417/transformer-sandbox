import os

import torch
import pandas as pd
from datasets import load_dataset
from pytest_mock import MockFixture
from transfer_classifier.augmentor.autoencoder_augmentor import AutoEncoderAugmentor
from transfer_classifier.dataset_preprocessor.amazon_review import AmazonReview
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class TestAutoEncoderAugmentor:
    def test_replace_words(self) -> None:
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        model = AutoModelForMaskedLM.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        augmentor = AutoEncoderAugmentor(
            model=model, tokenizer=tokenizer, replace_rate=0.2
        )

        text = "今日もいい天気で、花がきれいに咲いています"
        num_replaced, replaced = augmentor.replace_words(text, lang="ja")
        assert num_replaced > 0
        assert text != replaced

    def test_generate(self) -> None:
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        model = AutoModelForMaskedLM.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        augmentor = AutoEncoderAugmentor(
            model=model, tokenizer=tokenizer, replace_rate=0.3
        )

        review = AmazonReview(input_column="review_title", label_column="stars")
        samples = review.load("validation").select(range(10))

        augmenteds = augmentor.generate(samples, review)
        result = []
        for original, augmented in zip(samples, augmenteds):
            result.append(
                {
                    "original": original[review.input_column],
                    "augmented": augmented[review.input_column],
                    "stars": original[review.label_column],
                }
            )

        df = pd.DataFrame(result)
        # df.to_csv("autoencoder.csv", index=False)
