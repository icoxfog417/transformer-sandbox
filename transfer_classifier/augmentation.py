import copy
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from augmentor.augmentor import Augmentor
from augmentor.autoencoder_augmentor import AutoEncoderAugmentor
from augmentor.autoregressive_augmentor import AutoRegressiveAugmentor
from dataset_preprocessor.amazon_review import AmazonReview
from datasets import concatenate_datasets
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


def main(
    input_column: str = "review_title",
    augment_method: str = "autoencoder",
    num_run: int = 3,
    num_samples: int = 100,
    batch_size: int = 10,
    model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking",
    threshold: float = 0.6,
    replace_rate: float = 0.3,
    eval_frequency: int = 3,
) -> Tuple[PreTrainedModel, pd.DataFrame]:

    # Read data
    review = AmazonReview(input_column=input_column, label_column="stars", tokenizer=None, lang="ja")

    # Define evaluation setting
    dataset = review.load("train")
    validation_dataset = review.load("validation")

    validation_samples = None

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

    def create_augmentor(augment_method: str) -> Augmentor:

        if augment_method == "autoregressive":
            model_name = "rinna/japanese-gpt2-medium"
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
            augmentor = AutoRegressiveAugmentor(model=model, tokenizer=tokenizer, num_prompt=3)
        else:
            model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            augmentor = AutoEncoderAugmentor(model=model, tokenizer=tokenizer, replace_rate=replace_rate)
        return augmentor

    discriminator = None

    compares = []
    eval_steps = (num_samples / batch_size) // eval_frequency
    for i in range(num_run):
        # Define pretrained tokenizer and model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if validation_samples is None:
            review.tokenizer = tokenizer
            validation_samples = review.format(validation_dataset).select(range(256))

        samples = dataset.shuffle().select(range(num_samples))

        if i == 0:
            print(f"Number of samples is {len(samples)}")
        else:
            augmentor = create_augmentor(augment_method)
            augmenteds = augmentor.augment(samples, review, discriminator=discriminator, threshold=threshold)
            print(f"Number of samples is {len(samples)} and augmenteds is {len(augmenteds)}.")

            for sample, augmented in zip(samples, augmenteds):
                compares.append(
                    {
                        "num_trial": i,
                        "original": sample[review.input_column],
                        "augmented": augmented[review.input_column],
                        "stars": sample[review.label_column],
                    }
                )
            samples = concatenate_datasets([samples, augmenteds])

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
            eval_steps=eval_steps,
            seed=0,
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

        if i == 0:
            discriminator = copy.deepcopy(trainer.model)

    return (trainer.model, pd.DataFrame(compares))
