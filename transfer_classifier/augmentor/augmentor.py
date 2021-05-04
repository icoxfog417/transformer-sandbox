from typing import Tuple
from typing import Dict
from typing import Union
from typing import List
from typing import Any
from typing import Optional
from math import ceil
import random
import torch
import numpy as np
from datasets.arrow_dataset import Dataset
from datasets import concatenate_datasets
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding
from transfer_classifier.dataset_preprocessor.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)


class Augmentor:
    def __init__(self) -> None:
        self.__AUGMENTATION_VALID__ = "VALID"

    def augment(
        self,
        dataset: Dataset,
        preprocessor: ClassificationDatasetPreprocessor,
        num_trial: int = 1,
        discriminator: PreTrainedModel = None,
        threshold: float = 0.8,
    ) -> BatchEncoding:
        augmented_samples = None  # type: Optional[BatchEncoding]

        if discriminator is not None and preprocessor is None:
            raise Exception("To use discriminator, preprocessor should be required.")

        for i in range(num_trial):
            augmented = self.generate(dataset, preprocessor)
            if discriminator is not None and preprocessor is not None:
                matched = self.discriminate(
                    discriminator, preprocessor, dataset, augmented, threshold
                )

                def unmatched_to_invalid(
                    example: Dict[str, Any], index: int
                ) -> Dict[str, Any]:
                    example[self.__AUGMENTATION_VALID__] = (
                        True if index in matched else False
                    )
                    return example

                augmented = dataset.map(unmatched_to_invalid, with_indices=True)

            augmented = augmented.filter(lambda e: e[self.__AUGMENTATION_VALID__])
            if len(augmented) == 0:
                continue

            if augmented_samples is None:
                augmented_samples = augmented
            else:
                augmented_samples = concatenate_datasets([augmented_samples, augmented])

            if len(dataset) < len(augmented_samples):
                augmented_samples = augmented_samples.select(range(len(dataset)))
                break

        if augmented_samples is not None:
            augmented_samples = augmented_samples.remove_columns(
                [self.__AUGMENTATION_VALID__]
            )
            augmented_samples = augmented_samples.flatten_indices()

        return augmented_samples

    def generate(
        self, dataset: Dataset, preprocessor: ClassificationDatasetPreprocessor
    ) -> BatchEncoding:
        raise NotImplementedError("Augmentor subclass should implement augment_sample.")

    def discriminate(
        self,
        model: PreTrainedModel,
        preprocessor: ClassificationDatasetPreprocessor,
        original: Dataset,
        augmented: Dataset,
        threshold: float,
    ) -> List[int]:

        formatted_original = preprocessor.format(original)
        original_scores = self.predict(model, preprocessor, formatted_original)

        formatted_augmented = preprocessor.format(augmented)
        augmented_scores = self.predict(model, preprocessor, formatted_augmented)

        matched = []
        for i, original, augmented in zip(
            range(len(original)), original_scores, augmented_scores
        ):
            if (
                original["label"] == augmented["label"]
                and augmented["score"] >= threshold
            ):
                matched.append(i)

        return matched

    def predict(
        self,
        model: PreTrainedModel,
        preprocessor: ClassificationDatasetPreprocessor,
        examples: Dataset,
    ) -> List[Dict[str, Union[int, float]]]:
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)
        with torch.no_grad():  # type: ignore
            input_ids = examples["input_ids"].to(device)
            if "token_type_ids" in examples.column_names:
                token_type_ids = examples["token_type_ids"].to(device)
                outputs = model(input_ids, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids)

            predictions = outputs[0].cpu().numpy()

        scores = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
        return [
            {"label": model.config.id2label[item.argmax()], "score": item.max().item()}
            for item in scores
        ]
