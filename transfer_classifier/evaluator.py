from typing import Dict
from typing import Tuple
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from datasets import concatenate_datasets, load_from_disk
from amazon_review import AmazonReview
from augmentor import Augmentor

# Read data
review = AmazonReview(lang="en")

# Define pretrained tokenizer and model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_augment = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define evaluation setting
num_run = 2
num_samples = 100
dataset = review.load("train")
validation_dataset = review.format(review.load("validation"), tokenizer=tokenizer)

for i in range(num_run):
    samples = dataset.shuffle().select(range(num_samples))
    print(f"Number of samples is {len(samples)}")

    samples = review.format(samples, tokenizer)
    training_args = TrainingArguments(
        output_dir=f"./results/run{i + 1}",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=f"./logs/run{i + 1}",  # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        compute_metrics=compute_metrics,
        train_dataset=samples,  # training dataset
        eval_dataset=validation_dataset,  # evaluation dataset
    )

    trainer.train()
