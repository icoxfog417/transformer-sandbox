from typing import Dict
from typing import Tuple
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import EarlyStoppingCallback
from amazon_review import AmazonReview

# Read data
# About slice https://huggingface.co/docs/datasets/splits.html
review = AmazonReview(lang="ja")

# Define pretrained tokenizer and model
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = review.load("validation")

dataset = dataset.train_test_split(test_size=0.2)
dataset_train = review.format(dataset["train"], tokenizer)
dataset_validation = review.format(dataset["test"], tokenizer)


# Define Trainer parameters
def compute_metrics(eval: EvalPrediction) -> Dict[str, float]:
    pred, labels = eval
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Define Trainer
args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    seed=0,
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_validation,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()
