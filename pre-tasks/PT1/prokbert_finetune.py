import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# 1. LOAD, BALANCE, AND MERGE YOUR REAL EXPERIMENTAL CSV DATASETS
print("Loading experimental CSV datasets...")

# Load positive and negative data frames
df_pos = pd.read_csv("promoters_with_labels.csv")  # True promoters (Label 1)
df_neg = pd.read_csv("nonprom_with_labels.csv")    # Non-promoters (Label 0)

# Standardize the column names so they match perfectly before combining
df_pos.columns = ['sequence', 'label']
df_neg.columns = ['sequence', 'label']

# Convert all sequences to uppercase to ensure consistency
df_pos['sequence'] = df_pos['sequence'].str.upper()
df_neg['sequence'] = df_neg['sequence'].str.upper()

# ML Best Practice: Balance dataset classes so the model isn't biased
min_samples = min(len(df_pos), len(df_neg))
print(f"Balancing dataset: Selecting {min_samples} samples from each class.")

df_pos_balanced = df_pos.sample(n=min_samples, random_state=42)
df_neg_balanced = df_neg.sample(n=min_samples, random_state=42)

# Merge them together and shuffle the whole deck
df_combined = pd.concat([df_pos_balanced, df_neg_balanced], axis=0)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert your Pandas DataFrame straight into a Hugging Face Dataset object
dataset = Dataset.from_pandas(df_combined)

# Split into 80% training and 20% validation (better split for real datasets)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Dataset Split complete! Train size: {len(dataset['train'])} | Validation size: {len(dataset['test'])}")

# 2. Load the official ProkBERT Tokenizer
# ProkBERT doesn't read text; it uses an overlapping k-mer method to tokenize microbial DNA
print("Loading ProkBERT Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True, token="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")

# Tokenize our DNA sequences
def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=40)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Load ProkBERT-mini with a Binary Classification Head (Promoter vs Non-Promoter)
print("Loading ProkBERT-mini Model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "neuralbioinfo/prokbert-mini", 
    num_labels=2, 
    trust_remote_code=True,
    token="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"
)

# 4. Define Training parameters
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=torch.cuda.is_available(), # Uses GPU acceleration if available!
)

# 5. Initialize the Hugging Face Trainer engine
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 6. Run the fine-tuning loop!
print("Starting training loop...")
trainer.train()

print("Success! Saving fine-tuned model...")
trainer.save_model("./final_prokbert_promoter_model")
tokenizer.save_pretrained("./final_prokbert_promoter_model")