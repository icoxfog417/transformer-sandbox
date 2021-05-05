from pathlib import Path
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    T5Tokenizer,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from dataset_preprocessor.amazon_review import AmazonReview
from augmentor.augmentor import Augmentor
from augmentor.autoencoder_augmentor import AutoEncoderAugmentor
from augmentor.autoregressive_augmentor import AutoRegressiveAugmentor


def write_dataset(
    input_column: str = "review_title",
    augment_method: str = "autoencoder",
    save_folder: str = "experiments",
    range_from: int = 0,
    range_to: int = 5,
    num_samples: int = 100,
    model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking",
    replace_rate: float = 0.3,
    num_prompt: int = 3,
    max_length_factor: float = 3,
) -> List[pd.DataFrame]:

    # Read data
    review = AmazonReview(
        input_column=input_column, label_column="stars", tokenizer=None, lang="ja"
    )

    # Define evaluation setting
    dataset = review.load("train")

    def create_augmentor(augment_method: str) -> Augmentor:

        if augment_method == "autoregressive":
            model_name = "rinna/japanese-gpt2-medium"
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
            augmentor = AutoRegressiveAugmentor(
                model=model,
                tokenizer=tokenizer,
                num_prompt=num_prompt,
                max_length_factor=max_length_factor,
            )
        else:
            model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            augmentor = AutoEncoderAugmentor(
                model=model, tokenizer=tokenizer, replace_rate=replace_rate
            )
        return augmentor

    # Define pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    review.tokenizer = tokenizer

    dfs = []
    for i in range(range_from, range_to):
        print(f"Iteration {i}")
        samples = dataset.shuffle().select(range(num_samples))
        augmentor = create_augmentor(augment_method)
        augmenteds = augmentor.augment(samples, review)

        augmented_dataset = []
        for sample in samples:
            augmented_dataset.append(
                {
                    review.input_column: sample[review.input_column],
                    "stars": sample[review.label_column],
                    "kind": "original",
                }
            )

        for augmented in augmenteds:
            augmented_dataset.append(
                {
                    review.input_column: augmented[review.input_column],
                    "stars": augmented[review.label_column],
                    "kind": "augmented",
                }
            )

        path = Path(f"./{save_folder}")
        if not path.exists():
            path.mkdir()
        df = pd.DataFrame(augmented_dataset)
        file_name = f"{augment_method}_{i}.csv"
        print(
            f"Save {len(samples)} samples and {len(augmenteds)} augmented data to {file_name}."
        )
        df.to_csv(path.joinpath(file_name), index=False)
        dfs.append(df)

    return dfs


def train_experiment(
    input_column: str = "review_title",
    augment_method: str = "autoencoder",
    save_folder: str = "experiments",
    range_from: int = 0,
    range_to: int = 5,
    model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking",
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

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    review = AmazonReview(
        input_column=input_column, label_column="stars", tokenizer=tokenizer, lang="ja"
    )
    validation_dataset = review.load("validation")
    path = Path(f"./{save_folder}")

    without_augmentation = False
    for i in range(range_from, range_to):
        file_name = f"{augment_method}_{i}.csv"
        dataset = load_dataset("csv", data_files=str(path.joinpath(file_name)))["train"]
        validation_samples = review.format(validation_dataset.shuffle()).select(
            range(len(dataset))
        )

        for kind in ("original", "augmented"):
            if kind == "original":
                if not without_augmentation:
                    print("Show without augmentation")
                    samples = dataset.filter(lambda e: e["kind"] == kind)
                    without_augmentation = True
                else:
                    continue
            else:
                print(f"Iteration {i}")
                samples = dataset  # include all dataset

            samples = review.format(samples)
            if Path(f"./results/run{i}").exists():
                shutil.rmtree(f"./results/run{i}")

            training_args = TrainingArguments(
                output_dir=f"./results/run{i + 1}",  # output directory
                num_train_epochs=3,  # total number of training epochs
                load_best_model_at_end=True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=32,
                evaluation_strategy="steps",
                eval_steps=len(samples) // batch_size // eval_interval,
                logging_dir=f"./logs/run{i + 1}",  # directory for storing logs
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

    return trainer.model
