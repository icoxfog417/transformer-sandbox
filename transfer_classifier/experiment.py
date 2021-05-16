import shutil
from pathlib import Path
from typing import Dict

import augmentor as aug
import dataset_preprocessor as dp
import numpy as np
from augmented_dataset import AugmentedDataset
from dataset_preprocessor.classification_dataset_preprocessor import (
    ClassificationDatasetPreprocessor,
)
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
) -> ClassificationDatasetPreprocessor:
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


def make_directory_name(
    augment_method: str = "autoencoder",
    dataset_name: str = "amazon_review",
    input_column: str = "review_title",
    discriminator: bool = False,
) -> str:
    name = (
        f"{dataset_name}_{augment_method}_{input_column}{'_D' if discriminator else ''}"
    )
    return name


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
    preprocessor = load_dataset_preprocessor(
        dataset_name, input_column, label_column, max_length
    )

    dataset = preprocessor.load("train")

    discriminator_model = None
    if discriminator:
        directory = make_directory_name(
            augment_method, dataset_name, input_column, discriminator
        )
        samples = dataset.load_dataset()["train"].shuffle().select(range(num_samples))
        augmented = AugmentedDataset(path.joinpath(directory), str("model"))
        print(f"Save {num_samples} samples for discriminator to {directory}.")

        augmented.save_dataset(
            preprocessor,
            samples,
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
        augmentor = aug.AutoEncoderAugmentor(
            model=model, tokenizer=tokenizer, replace_rate=replace_rate
        )

    for i in range(range_from, range_to):
        print(f"Iteration {i}")
        directory = make_directory_name(
            augment_method, dataset_name, input_column, discriminator
        )
        augmented = AugmentedDataset(path.joinpath(directory), str(i))
        samples = dataset.shuffle().select(range(num_samples))
        augmenteds = augmentor.augment(
            dataset=samples,
            preprocessor=preprocessor,
            num_trial=num_trial,
            discriminator=discriminator_model,
            threshold=threshold,
        )
        augmented.save_dataset(
            preprocessor,
            samples,
            augmenteds,
            preprocessor.load("validation"),
            preprocessor.load("test"),
        )

        print(f"Save augmented dataset to {directory}.")


def train_experiment(
    augment_method: str = "autoencoder",
    dataset_name: str = "amazon_review",
    input_column: str = "review_title",
    discriminator: bool = False,
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
        recall = recall_score(y_true=labels, y_pred=pred, average="micro")
        precision = precision_score(y_true=labels, y_pred=pred, average="micro")
        f1 = f1_score(y_true=labels, y_pred=pred, average="micro")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    directory = make_directory_name(
        augment_method, dataset_name, input_column, discriminator
    )
    num_labels = 2
    if dataset_name == "livedoor":
        num_labels = dp.Livedoor.NUM_CLASS
    else:
        num_labels = dp.AmazonReview.NUM_CLASS

    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFICATION_MODEL_NAME, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL_NAME)

    path = Path(f"./{save_folder}")
    index = 0
    for index, sample_path in enumerate(
        [d for d in path.joinpath(directory).iterdir() if d.is_dir()]
    ):
        if index < range_from:
            continue
        elif index >= range_to:
            break
        augmented = AugmentedDataset(str(sample_path.parent), sample_path.name)
        samples = augmented.load_dataset()["train"]
        validation_samples = augmented.load_dataset()["validation"]
        validation_samples = augmented.format(
            dataset=validation_samples,
            tokenizer=tokenizer,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
        )

        print(f"Iteration {index}")
        for kind in ("original", "augmented"):
            if compare:
                if kind == "original":
                    samples = samples.filter(
                        lambda e: e[augmented._kind_column] == kind
                    )

            samples = augmented.format(
                dataset=samples,
                tokenizer=tokenizer,
                truncation=truncation,
                max_length=max_length,
                padding=padding,
            )
            if Path(f"./results/{directory}/{index}").exists():
                shutil.rmtree(f"./results/{directory}/{index}")

            training_args = TrainingArguments(
                output_dir=f"./results/{directory}/{index + 1}",  # output directory
                num_train_epochs=3,  # total number of training epochs
                load_best_model_at_end=True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=32,
                evaluation_strategy="steps",
                eval_steps=len(samples) // batch_size // eval_interval,
                logging_dir=f"./logs/{directory}/{index + 1}",  # directory for storing logs
            )

            trainer = Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                compute_metrics=compute_metrics,
                train_dataset=samples,  # training dataset
                eval_dataset=validation_samples,  # evaluation dataset
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

            trainer.train()

            if not compare:
                break

    return trainer.model
