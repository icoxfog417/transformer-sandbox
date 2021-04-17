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


class Augmentor:
    def __init__(
        self, lang: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
    ):
        self.lang = lang
        self.tagger = GenericTagger()  # Only Japanese support now
        self.model = model
        self.tokenizer = tokenizer

    def augment(
        self, dataset: Dataset, target: str, num_replace: int = 1
    ) -> BatchEncoding:
        def replace_words(example: Dict[str, Any]) -> Dict[str, Any]:
            _num_replace, text = self.replace_words(example[target], num_replace)
            example["augmented"] = True if _num_replace > 0 else False
            example[target] = text
            return example

        replaced = dataset.map(replace_words)
        augmented = replaced.filter(lambda e: e["augmented"])
        augmented = augmented.remove_columns(["augmented"])
        return augmented

    def replace_words(self, text: str, num_replace: int) -> Tuple[int, str]:
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

            if self.lang == "ja":
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
