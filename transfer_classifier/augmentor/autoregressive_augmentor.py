from typing import Any, Dict

from datasets.arrow_dataset import Dataset
from fugashi import GenericTagger
from transfer_classifier.augmentor.augmentor import Augmentor
from transfer_classifier.dataset_preprocessor.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)
from transformers import PreTrainedModel
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer


class AutoRegressiveAugmentor(Augmentor):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_prompt: int = 3,
        max_length_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.num_prompt = num_prompt
        self.max_length_factor = max_length_factor
        self.tagger = GenericTagger()  # Only Japanese support now

    def generate(
        self, dataset: Dataset, preprocessor: ClassificationDatasetPreprocessor
    ) -> BatchEncoding:
        def truncate_words(example: Dict[str, Any]) -> Dict[str, Any]:
            words = [
                token.surface
                for token in self.tagger(example[preprocessor.input_column])
            ]
            prompt = words[: self.num_prompt]
            if preprocessor.lang == "ja":
                _text = "".join(prompt)
            else:
                _text = " ".join(prompt)

            example[preprocessor.input_column] = _text
            example["original_length"] = len(example[preprocessor.input_column])
            return example

        truncateds = dataset.map(truncate_words)

        def attach_generated_words(
            example: Dict[str, Any], index: int
        ) -> Dict[str, Any]:

            formatted_truncated = self.tokenizer.encode(
                truncateds[index][preprocessor.input_column], return_tensors="pt"
            )

            # tutorial
            # https://huggingface.co/blog/how-to-generate
            max_length = int(
                truncateds[index]["original_length"] * self.max_length_factor
            )
            generated = self.model.generate(
                formatted_truncated,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                max_length=preprocessor.max_length,
            )

            _text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            example[preprocessor.input_column] = _text
            example[self.__AUGMENTATION_VALID__] = True
            return example

        generateds = dataset.map(attach_generated_words, with_indices=True)

        return generateds
