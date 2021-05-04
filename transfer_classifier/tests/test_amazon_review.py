from transfer_classifier.dataset_preprocessor.amazon_review import AmazonReview
from transformers import AutoTokenizer


class TestAmazonReview:
    def test_load(self) -> None:
        review = AmazonReview(input_column="review_title", label_column="stars", lang="ja")
        assert len(review.load("validation")) > 0

    def test_tokenize(self) -> None:
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        review = AmazonReview(
            input_column="review_title",
            label_column="stars",
            tokenizer=tokenizer,
            lang="ja",
        )
        dataset = review.load("validation")
        tokenized = review.tokenize(dataset)
        assert "input_ids" in tokenized.features
        assert "token_type_ids" in tokenized.features
        assert "attention_mask" in tokenized.features
        assert len(dataset) == len(tokenized)
        assert len(dataset) == len(tokenized["input_ids"])
        assert len(dataset) == len(tokenized["token_type_ids"])
        assert len(dataset) == len(tokenized["attention_mask"])

    def test_labels(self) -> None:
        review = AmazonReview(input_column="review_title", label_column="stars", lang="ja")
        dataset = review.load("validation", filter_medium_star=False)
        labeled = review.format_labels(dataset)
        assert "labels" in labeled.features
        assert len(dataset) == len(labeled["labels"])

        # star 1 = 0
        assert len(labeled.filter(lambda s: s["stars"] == 1)) > 0
        assert len(labeled.filter(lambda s: s["stars"] == 1)) == len(labeled.filter(lambda s: s["labels"] == 0))

        # star 5 = 1
        assert len(labeled.filter(lambda s: s["stars"] == 5)) > 0
        assert len(labeled.filter(lambda s: s["stars"] == 5)) == len(labeled.filter(lambda s: s["labels"] == 1))

        # star 5 = 1
        assert len(labeled.filter(lambda s: 1 < s["stars"] < 5)) > 0
        assert len(labeled.filter(lambda s: 1 < s["stars"] < 5)) == len(labeled.filter(lambda s: s["labels"] == -1))

    def test_format(self) -> None:
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        review = AmazonReview(
            input_column="review_title",
            label_column="stars",
            tokenizer=tokenizer,
            lang="ja",
        )
        dataset = review.load("validation")
        formatted = review.format(dataset)
        example = formatted[0]
        assert len(example) == 4
        assert "labels" in example
        assert "input_ids" in example
        assert "token_type_ids" in example
        assert "attention_mask" in example
        statistics = review.statistics(formatted)
        assert statistics["total"] > 0
        assert statistics["positive"] > 0
        assert statistics["negative"] > 0
        assert statistics["negative"] + statistics["positive"] == statistics["total"]
