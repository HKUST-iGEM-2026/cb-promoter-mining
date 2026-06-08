import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# 1. Create a tiny mock dataset so we can verify the code runs perfectly
data = {
    "sequence": [
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",  # Mock Promoter (Label 1)
        "TTTTGGCCAAAATTTTGGCCAAAATTTTGGCCAAAATTTT",  # Mock Promoter (Label 1)
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",  # Mock Non-Promoter (Label 0)
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"   # Mock Non-Promoter (Label 0)
    ],
    "label": [1, 1, 0, 0]
}
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Split into 50% training and 50% validation
dataset = dataset.train_test_split(test_size=0.5)

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