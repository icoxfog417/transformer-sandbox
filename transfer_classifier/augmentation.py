import copy
from typing import Dict

import numpy as np
from amazon_review import AmazonReview
from autoencoder_augmentor import AutoEncoderAugmentor
from datasets import concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

# Read data
review = AmazonReview(
    input_column="review_title", label_column="stars", tokenizer=None, lang="ja"
)

# Define evaluation setting
dataset = review.load("train")
validation_dataset = review.load("validation")

num_run = 2
num_samples = 128
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
validation_samples = None


def compute_metrics(eval: EvalPrediction) -> Dict[str, float]:
    pred, labels = eval
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def create_augmentor() -> AutoEncoderAugmentor:
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    augmentor = AutoEncoderAugmentor(model=model, tokenizer=tokenizer)
    return augmentor


discriminator = None

for i in range(num_run):
    # Define pretrained tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if validation_samples is None:
        review.tokenizer = tokenizer
        validation_samples = review.format(validation_dataset).select(range(256))
        review.tokenizer = tokenizer

    samples = dataset.shuffle().select(range(num_samples))

    if i == 0:
        print(f"Number of samples is {len(samples)}")
    else:
        augmentor = create_augmentor()
        augmented = augmentor.augment(samples, review, discriminator=discriminator)
        print(f"Number of samples is {len(samples)} and augmented is {len(augmented)}.")
        samples = concatenate_datasets([samples, augmented])

    samples = review.format(samples)
    training_args = TrainingArguments(
        output_dir=f"./results/run{i + 1}",  # output directory
        num_train_epochs=3,  # total number of training epochs
        save_strategy="no",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=8,
        seed=0,
        load_best_model_at_end=True,
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
        discriminator = copy.deepcopy(model)
