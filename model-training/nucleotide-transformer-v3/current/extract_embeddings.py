#!/usr/bin/env python3
import os
import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def main():
    model_dir = "./results_ntv3_full/final_model"
    model_id = "InstaDeepAI/NTv3_650M_pre"
    data_dir = "/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T15_ctx"
    
    print(f"Loading tokenizer from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print(f"Loading model from: {model_dir}")
    # Load model with output_hidden_states=True to extract embeddings
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, output_hidden_states=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load splits (adjust file names if your splits are named differently)
    splits = ["train", "val", "test"]
    data_files = {split: os.path.join(data_dir, f"{split}.parquet") for split in splits}
    
    dataset_dict = load_dataset("parquet", data_files=data_files)
    
    def extract_split(split_name):
        print(f"Processing {split_name} split...")
        ds = dataset_dict[split_name]
        
        seq_col = "sequence" if "sequence" in ds.column_names else ds.column_names[0]
        label_col = "label" if "label" in ds.column_names else (ds.column_names[1] if len(ds.column_names) > 1 else None)
        
        embeddings_list = []
        labels_list = []
        
        # Batch processing
        batch_size = 8
        sequences = ds[seq_col]
        labels = ds[label_col] if label_col else [0] * len(sequences)
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                inputs = tokenizer(
                    batch_seqs,
                    add_special_tokens=False,
                    padding="max_length",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids = inputs["input_ids"].to(device)
                attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Get hidden states from the last layer: (batch_size, seq_len, hidden_dim)
                hidden_states = outputs.hidden_states[-1]
                
                # Mean pooling over the sequence length, considering attention mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                embeddings_list.append(mean_embeddings)
                labels_list.extend(batch_labels)
                
        X = np.concatenate(embeddings_list, axis=0)
        y = np.array(labels_list)
        
        np.save(f"{split_name}_X.npy", X)
        np.save(f"{split_name}_y.npy", y)
        print(f"Saved {split_name}_X.npy with shape {X.shape} and {split_name}_y.npy")

    for split in dataset_dict.keys():
        extract_split(split)
        
    print("Embedding extraction complete!")

if __name__ == "__main__":
    main()
