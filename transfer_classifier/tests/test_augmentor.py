from transformers import AutoModelForMaskedLM, AutoTokenizer
from transfer_classifier.augmentor import Augmentor
from transfer_classifier.amazon_review import AmazonReview


class TestAugmentor:
    def test_replace_words(self) -> None:
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        model = AutoModelForMaskedLM.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        augmentor = Augmentor(lang="ja", model=model, tokenizer=tokenizer)

        text = "今日もいい天気で、花がきれいに咲いています"
        num_replaced, replaced = augmentor.replace_words(text, num_replace=1)
        assert num_replaced > 0
        assert text != replaced

    def test_augment(self) -> None:
        review = AmazonReview(lang="ja")
        samples = review.load("validation").select(range(10))

        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        model = AutoModelForMaskedLM.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        augmentor = Augmentor(lang="ja", model=model, tokenizer=tokenizer)
        augmented = augmentor.augment(samples, "review_title")
        assert len(augmented) > 0
        assert "augmented" not in augmented.features
