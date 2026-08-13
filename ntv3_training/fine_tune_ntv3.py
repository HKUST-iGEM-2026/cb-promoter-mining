#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def main():
    model_id = "InstaDeepAI/NTv3_650M_pre"
    output_dir = "./results_ntv3_full"
    
    print(f"Loading model backbone & tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    
    # Load actual pre-split parquet datasets from the pipeline results path
    print("Loading datasets from cluster parquet files...")
    data_dir = "/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T15_ctx"
    
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(data_dir, "train.parquet"),
            "validation": os.path.join(data_dir, "val.parquet")
        }
    )
    
    # Tokenization function
    def tokenize_function(examples):
        # Adjust column name if your parquet uses something other than 'sequence'
        seq_col = "sequence" if "sequence" in examples else list(examples.keys())[0]
        tokenized = tokenizer(
            examples[seq_col],
            add_special_tokens=False,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        report_to="none"
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    print("Starting full fine-tuning...")
    trainer.train()
    
    trainer.save_model(os.path.join(output_dir, "final_model"))
    print("Full fine-tuning complete!")

if __name__ == "__main__":
    main()
