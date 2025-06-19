import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPU:" if torch.cuda.is_available() else "Using CPU:", device)
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# Load full training data and separate test set
full_train_df = pd.read_csv("dataset/training_data.csv", sep="\t", header=None, names=["label", "headline"])
test_df = pd.read_csv("dataset/testing_data.csv", sep="\t", header=None, names=["label", "headline"])

# Validate columns
assert "headline" in full_train_df.columns and "label" in full_train_df.columns
assert "headline" in test_df.columns and "label" in test_df.columns

# Rename columns
full_train_df = full_train_df.rename(columns={"headline": "text", "label": "labels"})
test_df = test_df.rename(columns={"headline": "text", "label": "labels"})

# Split 80% train / 20% eval
train_df, eval_df = train_test_split(full_train_df, test_size=0.2, random_state=42)

# Tokenizer
if os.path.isdir("saved-model"):
    tokenizer = AutoTokenizer.from_pretrained("saved-model")
else:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize and convert to Datasets
train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

eval_dataset = Dataset.from_pandas(eval_df).map(tokenize_function, batched=True)
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
if os.path.isdir("saved-model"):
    model = AutoModelForSequenceClassification.from_pretrained("saved-model")
else:
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.save_pretrained("saved-model")
    tokenizer.save_pretrained("saved-model")

model.to(device)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1,
    logging_steps=50
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save fine-tuned model
trainer.model.save_pretrained("saved-model")
tokenizer.save_pretrained("saved-model")

# Prepare test dataset
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Predict
preds = trainer.predict(test_dataset)
output_labels = preds.predictions.argmax(-1)

# Write predictions
test_df["label"] = output_labels
test_df = test_df.drop(columns=["text"])
test_df.to_csv("predictions.csv", index=False)

# Print evaluation metrics on test
print("Estimated Accuracy:", compute_metrics(preds))
