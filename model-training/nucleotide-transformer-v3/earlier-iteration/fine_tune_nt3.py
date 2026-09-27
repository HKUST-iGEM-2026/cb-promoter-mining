import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, roc_auc_score

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(
    PROMOTER_REPO, "infrastructure", "T3", "snakemake_scaffold", "results")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # If model outputs tuple of logits, grab the first element
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    probs = torch.softmax(torch.tensor(predictions), dim=-1).numpy()[:, 1]
    preds = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
        
    return {
        "accuracy": acc,
        "roc_auc": auc
    }

def main():
    print("=== Loading Parquet Datasets ===", flush=True)
    data_dir = os.path.join(PROMOTER_RESULTS, "T15_ctx")
    
    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    val_df = pd.read_parquet(f"{data_dir}/val.parquet")
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")

    # Convert pandas dataframes to Hugging Face datasets
    # Assumes columns are named 'sequence' (or 'text') and 'label'
    seq_col = 'sequence' if 'sequence' in train_df.columns else 'text'
    
    train_dataset = Dataset.from_pandas(train_df[[seq_col, 'label']].rename(columns={seq_col: 'text'}))
    val_dataset = Dataset.from_pandas(val_df[[seq_col, 'label']].rename(columns={seq_col: 'text'}))
    test_dataset = Dataset.from_pandas(test_df[[seq_col, 'label']].rename(columns={seq_col: 'text'}))

    # Load NTv3 Tokenizer and Model
    # Using the pre-trained backbone checkpoint variant suitable for fine-tuning
    model_id = "InstaDeepAI/NTv3_650M_pre" # Adjust variant size if needed (e.g., 100M_pre)
    print(f"\nLoading tokenizer and model from {model_id}...", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        trust_remote_code=True,
        stem_compute_dtype='bfloat16',
        down_convolution_compute_dtype='bfloat16'
    )

    # Tokenization function (NTv3 requires lengths matching architecture constraints, e.g. multiple of 128)
    def tokenize_function(examples):
        # Adjust max_length based on your context windows (e.g., 512 or 1024)
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    print("Tokenizing datasets...", flush=True)
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Define training arguments optimized for cluster GPUs
    training_args = TrainingArguments(
        output_dir="results/nt3_finetuned",
        learning_rate=2e-5,
        per_device_train_batch_size=4,   # Adjust lower if encountering Out-Of-Memory (OOM) errors
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        logging_steps=50,
        fp16=False,
        bf16=True,                       # Use bfloat16 for stable transformer training on cluster GPUs
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print("\n=== Starting Fine-Tuning ===", flush=True)
    trainer.train()

    print("\n=== Evaluating on Test Set ===", flush=True)
    test_results = trainer.evaluate(tokenized_test)
    print(f"Test Results: {test_results}", flush=True)
    
    # Save final model weights
    trainer.save_model("results/nt3_finetuned_best")
    print("Fine-tuning complete and best model saved successfully!", flush=True)

if __name__ == "__main__":
    main()
