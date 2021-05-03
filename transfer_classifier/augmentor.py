from typing import Tuple
from typing import Dict
from typing import Any
from typing import Optional
from math import ceil
import random
from fugashi import GenericTagger
import torch
from datasets.arrow_dataset import Dataset
from datasets import concatenate_datasets
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding
from transfer_classifier.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)


class Augmentor:
    def __init__(self):
        self.__AUGMENTATION_VALID__ = "VALID"

    def augment(
        self,
        dataset: Dataset,
        preprocessor: ClassificationDatasetPreprocessor,
        num_trial: int = 3,
        discriminator: PreTrainedModel = None,
        threshold: float = 0.8,
    ) -> BatchEncoding:
        augmented_samples = None

        if discriminator is not None and preprocessor is None:
            raise Exception("To use discriminator, preprocessor should be required.")

        for i in range(num_trial):
            augmented = self.generate(dataset, preprocessor)
            if discriminator is not None and preprocessor is not None:
                formatted = preprocessor.format(augmented)
                prediction = discriminator.predict(formatted)
                over_threshold_index = torch.where(prediction >= threshold)
                augmented = augmented.select(over_threshold_index)

            augmented = augmented.filter(lambda e: e[self.__AUGMENTATION_VALID__])
            if augmented_samples is None:
                augmented_samples = augmented
            else:
                augmented_samples = concatenate_datasets([augmented_samples, augmented])

            if len(dataset) == len(augmented_samples):
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
