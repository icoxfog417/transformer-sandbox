from transformers import AutoTokenizer
from finetune import AmazonReview


class TestTransfer():

    def test_load(self):
        review = AmazonReview(lang="ja")
        assert len(review.load("validation")) > 0

    def test_tokenize(self):
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        review = AmazonReview(lang="ja")
        dataset = review.load("validation")
        tokenized = review.tokenize(dataset, tokenizer)
        assert "input_ids" in tokenized.features
        assert "token_type_ids" in tokenized.features
        assert "attention_mask" in tokenized.features
        assert len(dataset) == len(tokenized)
        assert len(dataset) == len(tokenized["input_ids"])
        assert len(dataset) == len(tokenized["token_type_ids"])
        assert len(dataset) == len(tokenized["attention_mask"])

    def test_labels(self):
        review = AmazonReview(lang="ja")
        dataset = review.load("validation")
        labeled = review.labels(dataset)
        assert "labels" in labeled.features
        assert len(dataset) == len(labeled["labels"])

        # star 1 = 0
        assert len(labeled.filter(lambda s: s["stars"] == 1)) > 0
        assert len(labeled.filter(lambda s: s["stars"] == 1)) ==\
               len(labeled.filter(lambda s: s["labels"] == 0))

        # star 5 = 1
        assert len(labeled.filter(lambda s: s["stars"] == 5)) > 0
        assert len(labeled.filter(lambda s: s["stars"] == 5)) ==\
               len(labeled.filter(lambda s: s["labels"] == 1))

        # star 5 = 1
        assert len(labeled.filter(lambda s: 1 < s["stars"] < 5)) > 0
        assert len(labeled.filter(lambda s: 1 < s["stars"] < 5)) ==\
               len(labeled.filter(lambda s: s["labels"] == -1))

    def test_format(self):
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        review = AmazonReview(lang="ja")
        dataset = review.load("validation")
        formatted = review.format(dataset, tokenizer)
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
