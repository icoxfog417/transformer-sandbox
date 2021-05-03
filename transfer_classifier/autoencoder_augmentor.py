from typing import Union
from typing import Tuple
from typing import Dict
from typing import Any
from typing import Optional
from math import ceil
import random
from fugashi import GenericTagger
import torch
from datasets.arrow_dataset import Dataset
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding
from transfer_classifier.augmentor import Augmentor
from transfer_classifier.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)


class AutoEncoderAugmentor(Augmentor):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tagger = GenericTagger()  # Only Japanese support now

    def generate(
        self, dataset: Dataset, preprocessor: ClassificationDatasetPreprocessor
    ) -> BatchEncoding:
        num_replace = 1

        def replace_words(example: Dict[str, Any]) -> Dict[str, Any]:
            _num_replace, text = self.replace_words(
                preprocessor.input_column, num_replace, preprocessor.lang
            )
            example[self.__AUGMENTATION_VALID__] = True if _num_replace > 0 else False
            example[preprocessor.input_column] = text
            return example

        replaced = dataset.map(replace_words)
        return replaced

    def replace_words(self, text: str, num_replace: int, lang: str) -> Tuple[int, str]:
        _text = text
        replaced = []

        for i in range(num_replace):
            words = []
            replace_indexes = []
            for i, w in enumerate(self.tagger(_text)):
                words.append(w.surface)
                if w.feature[0] in ("名詞", "動詞", "形容詞", "副詞") and i not in replaced:
                    replace_indexes.append(i)

            if len(replace_indexes) == 0:
                break

            index = random.choice(replace_indexes)
            words[index] = self.tokenizer.mask_token

            if lang == "ja":
                _text = "".join(words)
            else:
                _text = " ".join(words)

            encoded = self.tokenizer.encode(_text, return_tensors="pt")
            mask_token_index = torch.where(encoded == self.tokenizer.mask_token_id)[1]
            logits = self.model(encoded).logits
            masked_token_logits = logits[0, mask_token_index, :]

            top_ten_tokens = (
                torch.topk(masked_token_logits, 10, dim=1).indices[0].tolist()
            )
            random.shuffle(top_ten_tokens)
            for token in top_ten_tokens:
                _text = _text.replace(
                    self.tokenizer.mask_token,
                    self.tokenizer.decode([token]),
                )
                if text != _text:
                    break

            if text != _text:
                replaced.append(index)

        return (len(replaced), _text)
