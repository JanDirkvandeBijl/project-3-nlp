import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Set CUDA_VISIBLE_DEVICES to use GPU 3 and 4
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"  # Use GPU 3 and 4

# Use multiple GPUs (cuda:2, cuda:3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPUs:", os.environ["CUDA_VISIBLE_DEVICES"] if torch.cuda.is_available() else "Using CPU:", device)
if torch.cuda.is_available():
    print("CUDA device(s):", torch.cuda.get_device_name(0))  # Print first GPU name (cuda:0)

# Load training and test data
full_train_df = pd.read_csv("dataset/training_data.csv", sep="\t", header=None, names=["label", "headline"])
test_df = pd.read_csv("dataset/testing_data.csv", sep="\t", header=None, names=["label", "headline"])

# Check columns
assert "headline" in full_train_df.columns and "label" in full_train_df.columns
assert "headline" in test_df.columns and "label" in test_df.columns

# Rename columns
full_train_df = full_train_df.rename(columns={"headline": "text", "label": "labels"})
test_df = test_df.rename(columns={"headline": "text", "label": "labels"})

# Split train/eval 80/20
train_df, eval_df = train_test_split(full_train_df, test_size=0.2, random_state=42)

# Define the directory for saving the model and tokenizer
persistent_directory = "./model_directory"  # Change this path if needed
os.makedirs(persistent_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Load tokenizer
if os.path.exists(os.path.join(persistent_directory, "config.json")):
    # Load saved model and tokenizer if the configuration exists
    tokenizer = AutoTokenizer.from_pretrained(persistent_directory)
    model = AutoModelForSequenceClassification.from_pretrained(persistent_directory)
    print("Loaded model from saved directory.")
else:
    # Initialize new model and tokenizer, then save them for later use
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Save the model and tokenizer properly, ensure config.json is included
    model.save_pretrained(persistent_directory)  # Save model weights and config
    tokenizer.save_pretrained(persistent_directory)  # Save tokenizer files
    print("Initialized a new model and saved it.")

# Move model to the specified device (GPU or CPU)
model.to(device)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

eval_dataset = Dataset.from_pandas(eval_df).map(tokenize_function, batched=True)
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Metrics calculation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Check for None or length mismatch
    if labels is None or preds is None:
        print("Labels or Predictions are None!")
        return {"accuracy": 0, "recall": 0}

    if len(labels) != len(preds):
        print(f"Length mismatch: labels={len(labels)}, preds={len(preds)}")
        return {"accuracy": 0, "recall": 0}

    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='binary')  # Use recall for binary classification
    return {"accuracy": acc, "recall": recall}

# TrainingArguments with best model tracking and early stopping
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_recall",  # This will use recall for best model selection
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,  # max limit — early stopping will halt sooner if needed
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1,
    logging_steps=50
)

# Trainer with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# Train with early stopping; best model is restored
trainer.train()

# Save best model
trainer.model.save_pretrained(persistent_directory)
tokenizer.save_pretrained(persistent_directory)

# Prepare test set
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Predict
preds = trainer.predict(test_dataset)
output_labels = preds.predictions.argmax(-1)

# Export predictions
test_df["label"] = output_labels
if "text" in test_df.columns:
    test_df = test_df.drop(columns=["text"])
test_df.to_csv("predictions.csv", index=False)

# Print final results
print("Estimated Accuracy:", compute_metrics(preds))
