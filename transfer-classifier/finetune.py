import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import EarlyStoppingCallback


class AmazonReview():

    def __init__(self, lang: str = "ja"):
        self.lang = lang

    def load(self, split: str, shuffle=True):
        dataset = load_dataset("amazon_reviews_multi", self.lang, split=split)
        if shuffle:
            return dataset.shuffle()
        else:
            return dataset

    def tokenize(self, dataset, tokenizer, batched=True):

        def encode(examples):
            tokenized = tokenizer(examples["review_title"], truncation=True, max_length=512, padding="max_length")
            return tokenized

        return dataset.map(encode, batched=batched)

    def labels(self, dataset, batched=True):

        def convert_star(star):
            if 1 < star < 5:
                return -1
            elif star == 1:
                return 0
            else:
                return 1

        def encode(examples):
            labels = {"labels": np.array([convert_star(s) for s in examples["stars"]])}
            return labels

        return dataset.map(encode, batched=batched)

    def format(self, dataset, tokenizer, batched=True):
        tokenized = self.tokenize(dataset, tokenizer, batched)
        labeled = self.labels(tokenized, batched)
        filtered = labeled.filter(lambda example: example["labels"] >= 0)
        filtered.set_format(type="torch",
                            columns=["input_ids",
                                     "token_type_ids",
                                     "attention_mask",
                                     "labels"])
        return filtered

    def statistics(self, formatted):
        positives = len([e for e in formatted if e["labels"].item() == 1])
        negatives = len([e for e in formatted if e["labels"].item() == 0])

        return {
            "total": len(formatted),
            "positive": positives,
            "negative": negatives,
        }


# Read data
# About slice https://huggingface.co/docs/datasets/splits.html
review = AmazonReview(lang="ja")

# Define pretrained tokenizer and model
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset_train = review.format(review.load("train[:20%]"), tokenizer)
dataset_validation = review.format(review.load("validation"), tokenizer)


# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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

# ----- 3. Predict -----#
# Load test data
model_path = "output/checkpoint-3"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
dataset_test = review.format(review.load("test"), tokenizer)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(dataset_test)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
