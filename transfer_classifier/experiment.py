import shutil
from pathlib import Path
from typing import Dict

import augmentor as aug
import dataset_preprocessor as dp
import numpy as np
import pandas as pd
from augmented_dataset import AugmentedDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)

CLASSIFICATION_MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"


def load_dataset_preprocessor(
    dataset_name: str, input_column: str, label_column: str, max_length: int = 128
) -> dp.ClassificationDatasetPreprocessor:
    # Read data
    if dataset_name == "livedoor":
        preprocessor = dp.Livedoor(
            input_column=input_column,
            label_column=label_column,
            tokenizer=None,
            max_length=max_length,
            lang="ja",
        )
    else:
        preprocessor = dp.AmazonReview(
            input_column=input_column,
            label_column=label_column,
            tokenizer=None,
            max_length=max_length,
            lang="ja",
        )

    return preprocessor


def write_dataset(
    augment_method: str = "autoencoder",
    dataset_name: str = "amazon_review",
    input_column: str = "review_title",
    label_column: str = "stars",
    max_length: int = 128,
    save_folder: str = "experiments",
    num_samples: int = 100,
    range_from: int = 0,
    range_to: int = 5,
    num_trial: int = 2,
    replace_rate: float = 0.3,
    num_prompt: int = 3,
    max_length_factor: float = 1.1,
    discriminator: bool = False,
    threshold: float = 0.6,
) -> None:

    path = Path(f"./{save_folder}")
    if not path.exists():
        path.mkdir()

    # Define evaluation setting
    preprocessor = load_dataset_preprocessor(dataset_name, input_column, label_column, max_length)

    dataset = preprocessor.load("train")

    discriminator_model = None
    if discriminator:
        directory = f"{dataset_name}_{augment_method}_{input_column}_{'D' if discriminator else ''}_model"
        dataset = AugmentedDataset(path, directory)
        print(f"Save {num_samples} samples for discriminator to {directory}.")

        dataset.save_dataset(
            preprocessor,
            dataset.shuffle().select(range(num_samples)),
            [],
            preprocessor.load("validation"),
            preprocessor.load("test"),
        )
        discriminator_model = train_experiment(
            directory,
            save_folder=save_folder,
            range_from=0,
            range_to=1,
            compare=False,
            max_length=max_length,
        )

    # Create Augmentor
    if augment_method == "autoregressive":
        model_name = "rinna/japanese-gpt2-medium"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        augmentor = aug.AutoRegressiveAugmentor(
            model=model,
            tokenizer=tokenizer,
            num_prompt=num_prompt,
            max_length_factor=max_length_factor,
        )
    else:
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        augmentor = aug.AutoEncoderAugmentor(model=model, tokenizer=tokenizer, replace_rate=replace_rate)

    for i in range(range_from, range_to):
        print(f"Iteration {i}")
        directory = f"{dataset_name}_{augment_method}_{input_column}_{'D' if discriminator else ''}_{i}"
        dataset = AugmentedDataset(path, directory)
        samples = dataset.shuffle().select(range(num_samples))
        augmenteds = augmentor.augment(
            dataset=samples,
            preprocessor=preprocessor,
            num_trial=num_trial,
            discriminator=discriminator_model,
            threshold=threshold,
        )
        dataset.save_dataset(
            preprocessor,
            samples,
            augmenteds,
            preprocessor.load("validation"),
            preprocessor.load("test"),
        )

        print(f"Save dataset to {directory}.")


def train_experiment(
    directory: str,
    save_folder: str = "experiments",
    range_from: int = 0,
    range_to: int = 5,
    compare: bool = True,
    truncation: bool = True,
    max_length: int = 512,
    padding: str = "max_length",
    batch_size: int = 10,
    eval_interval: int = 3,
) -> PreTrainedModel:
    def compute_metrics(eval: EvalPrediction) -> Dict[str, float]:
        pred, labels = eval
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    naming = directory.split("_")
    num_labels = 2
    if naming[0] == "livedoor":
        num_labels = dp.Livedoor.NUM_CLASS
    else:
        num_labels = dp.AmazonReview.NUM_CLASS

    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_MODEL_NAME, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL_NAME)

    path = Path(f"./{save_folder}")
    dataset = AugmentedDataset(path, directory)
    validation = dataset.load_dataset()["validation"]
    for i in range(range_from, range_to):
        samples = dataset.load_dataset()["train"]

        print(f"Iteration {i}")
        for kind in ("original", "augmented"):
            if not compare:
                samples = dataset
            else:
                if kind == "original":
                    samples = dataset.filter(lambda e: e[dataset._kind_column] == kind)
                else:
                    samples = dataset  # include all dataset

            samples = dataset.format(
                dataset=samples,
                tokenizer=tokenizer,
                truncation=truncation,
                max_length=max_length,
                padding=padding,
            )
            if Path(f"./results/{directory}/run_{i}").exists():
                shutil.rmtree(f"./results/{directory}/run_{i}")

            training_args = TrainingArguments(
                output_dir=f"./results/{directory}/run_{i + 1}",  # output directory
                num_train_epochs=3,  # total number of training epochs
                load_best_model_at_end=True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=32,
                evaluation_strategy="steps",
                eval_steps=len(samples) // batch_size // eval_interval,
                logging_dir=f"./logs/{directory}/log_{i + 1}",  # directory for storing logs
            )

            trainer = Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                compute_metrics=compute_metrics,
                train_dataset=samples,  # training dataset
                eval_dataset=validation,  # evaluation dataset
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

            trainer.train()

            if not compare:
                break

    return trainer.model
