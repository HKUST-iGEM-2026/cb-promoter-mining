import torch
import numpy as np
import pandas as pd
from evo2 import Evo2

# Load model
evo2_model = Evo2('evo2_7b')
layer_name = 'blocks.28.mlp.l3'

data_dir = '/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T15_ctx'
out_dir = '/project/igem2026/xzhangjj/embeddings'

import os
os.makedirs(out_dir, exist_ok=True)

for split in ['train', 'val', 'test']:
    print(f'\nProcessing {split}...')
    df = pd.read_parquet(f'{data_dir}/{split}.parquet')
    sequences = df['sequence'].tolist()
    y = df['label'].values

    X = []
    for i, seq in enumerate(sequences):
        input_ids = torch.tensor(
            evo2_model.tokenizer.tokenize(seq),
            dtype=torch.int,
        ).unsqueeze(0).to('cuda:0')

        _, embeddings = evo2_model(
            input_ids, return_embeddings=True, layer_names=[layer_name]
        )

        vec = embeddings[layer_name].float().mean(dim=1).squeeze(0).cpu().numpy()
        X.append(vec)

        if i % 500 == 0:
            print(f'  {i}/{len(sequences)}')

    X = np.array(X)
    np.save(f'{out_dir}/{split}_X.npy', X)
    np.save(f'{out_dir}/{split}_y.npy', y)
    print(f'  Done. X shape: {X.shape}')

print('\nAll splits complete.')