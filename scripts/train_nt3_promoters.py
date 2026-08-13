import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import optax
from flax import linen as nn

class PromoterClassifier(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return jnp.squeeze(x, axis=-1)

def load_parquet_split(split_name):
    path = f"/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T15_ctx/{split_name}.parquet"
    print(f"Loading {split_name} split from {path}...")
    return pd.read_parquet(path)

def main():
    print("=== NTv3 Promoter Mining Full Training Loop ===")
    print(f"JAX active devices: {jax.devices()}")

    # Configuration
    model_ckpt = "InstaDeepAI/NTv3_8M_pre"
    batch_size = 8
    max_length = 256
    learning_rate = 1e-4

    # Load Data
    train_df = load_parquet_split("train")
    print(f"Training dataset size: {len(train_df)}")

    # Load Tokenizer & Backbone
    print(f"Loading NTv3 model and tokenizer from {model_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(model_ckpt, trust_remote_code=True)

    # Initialize Classifier Head and Optimizer
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((batch_size, 384)) # NTv3 8M hidden dimension size check or adapt dynamically
    
    # Get hidden dimension dynamically from config if needed, or infer from a test pass
    sample_texts = train_df['sequence'].head(batch_size).tolist()
    sample_labels = train_df['label'].head(batch_size).values.astype(np.float32)

    encodings = tokenizer(sample_texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    outputs = backbone(input_ids=jnp.array(encodings["input_ids"]), attention_mask=jnp.array(encodings["attention_mask"]))
    hidden_dim = outputs.last_hidden_state.shape[-1]

    classifier = PromoterClassifier()
    params = classifier.init(key, jnp.ones((batch_size, hidden_dim)))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_inputs, batch_masks, batch_labels):
        def loss_fn(p):
            outputs = backbone(input_ids=batch_inputs, attention_mask=batch_masks)
            hidden_states = outputs.last_hidden_state
            
            # Mean pooling
            mask_expanded = jnp.expand_dims(batch_masks, -1)
            pooled = jnp.sum(hidden_states * mask_expanded, axis=1) / jnp.maximum(jnp.sum(mask_expanded, axis=1), 1e-9)
            
            logits = classifier.apply(p, pooled)
            loss = optax.sigmoid_binary_cross_entropy(logits, batch_labels).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Running initial test training step on GPU...")
    batch_inputs = jnp.array(encodings["input_ids"])
    batch_masks = jnp.array(encodings["attention_mask"])
    batch_labels = jnp.array(sample_labels)

    params, opt_state, loss = train_step(params, opt_state, batch_inputs, batch_masks, batch_labels)
    print(f"Training step successful! Initial loss: {loss:.4f}")

if __name__ == "__main__":
    main()
