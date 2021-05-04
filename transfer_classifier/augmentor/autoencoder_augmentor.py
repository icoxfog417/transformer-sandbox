import random
from math import ceil
from typing import Any, Dict, Tuple, Union

import torch
from datasets.arrow_dataset import Dataset
from fugashi import GenericTagger
from transfer_classifier.augmentor.augmentor import Augmentor
from transfer_classifier.dataset_preprocessor.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)
from transformers import PreTrainedModel
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer


class AutoEncoderAugmentor(Augmentor):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        replace_rate: Union[int, float] = 0.15,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.replace_rate = replace_rate
        self.tagger = GenericTagger()  # Only Japanese support now

    def generate(self, dataset: Dataset, preprocessor: ClassificationDatasetPreprocessor) -> BatchEncoding:
        def replace_words(example: Dict[str, Any]) -> Dict[str, Any]:
            _num_replaced, text = self.replace_words(example[preprocessor.input_column], preprocessor.lang)
            example[self.__AUGMENTATION_VALID__] = True if _num_replaced > 0 else False
            example[preprocessor.input_column] = text
            return example

        replaced = dataset.map(replace_words)
        return replaced

    def replace_words(self, text: str, lang: str) -> Tuple[int, str]:
        _text = text

        tokens = list(self.tagger(_text))
        indexes = []
        if isinstance(self.replace_rate, int):
            indexes = random.sample(range(len(tokens)), 1)
        else:
            indexes = random.sample(range(len(tokens)), ceil(len(tokens) * self.replace_rate))

        for i in indexes:
            words = [token.surface if j != i else self.tokenizer.mask_token for j, token in enumerate(tokens)]

            if lang == "ja":
                _text = "".join(words)
            else:
                _text = " ".join(words)

            encoded = self.tokenizer.encode(_text, return_tensors="pt")
            mask_token_index = torch.where(encoded == self.tokenizer.mask_token_id)[1]
            logits = self.model(encoded).logits
            masked_token_logits = logits[0, mask_token_index, :]

            top_ten_tokens = torch.topk(masked_token_logits, 10, dim=1).indices[0].tolist()
            random.shuffle(top_ten_tokens)
            for token in top_ten_tokens:
                _text = _text.replace(
                    self.tokenizer.mask_token,
                    self.tokenizer.decode([token]),
                )
                if text != _text:
                    break

        return (len(indexes), _text)
