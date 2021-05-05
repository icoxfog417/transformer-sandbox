import pandas as pd
from transformers import AutoModelForCausalLM, T5Tokenizer
from transfer_classifier.augmentor.autoregressive_augmentor import (
    AutoRegressiveAugmentor,
)
from transfer_classifier.dataset_preprocessor.amazon_review import AmazonReview


class TestAutoRegressiveAugmentor:
    def test_generate(self) -> None:
        model_name = "rinna/japanese-gpt2-medium"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        augmentor = AutoRegressiveAugmentor(
            model=model, tokenizer=tokenizer, num_prompt=3
        )

        review = AmazonReview(input_column="review_title", label_column="stars")
        samples = review.load("validation").select(range(3))

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
        df.to_csv("autoregressive.csv", index=False)
        assert df is not None
