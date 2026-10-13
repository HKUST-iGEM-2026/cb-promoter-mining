#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(PROMOTER_REPO, "workflow", "results")


def main():
    model_dir = "./results_ntv3_full/final_model"
    model_id = "InstaDeepAI/NTv3_650M_pre"
    data_dir = os.path.join(PROMOTER_RESULTS, "T15_ctx")
    
    print(f"Loading tokenizer from base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print(f"Loading fine-tuned model weights from: {model_dir}")
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print("Loading test dataset...")
    test_dataset = load_dataset("parquet", data_files={"test": os.path.join(data_dir, "test.parquet")})["test"]
    
    def tokenize_function(examples):
        seq_col = "sequence" if "sequence" in examples else list(examples.keys())[0]
        # Tokenize sequences
        input_ids = tokenizer(
            examples[seq_col],
            add_special_tokens=False,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        
        # Create standard attention mask and labels
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    print("Tokenizing test dataset...")
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    dataloader = DataLoader(tokenized_test, batch_size=4)
    
    total_loss = 0.0
    print("Running evaluation on test set...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Complete! Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
