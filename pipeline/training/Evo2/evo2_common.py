#!/usr/bin/env python3
"""
evo2_common.py — shared batched Evo2 embedding, used by extract_embeddings_evo2.py (training) and
evo2_scan_restricted.py (deployment).

WHY BATCHING: the original one-at-a-time loop is ~1-2 seq/s — fine for ~47k training sequences, far
too slow for a restricted genome scan (~10^5-10^6 windows). Batching equal-length windows through one
forward is the difference between hours and weeks for the scan.

⚠️ Vortex may enforce max_batch_size=1 (the 7B config logs `'max_batch_size': 1`). So embed_sequences
TRIES a batched forward and, if it raises, DETECTS that once and falls back to per-item for the rest of
the run — correct either way, just slower when unbatched. If you hit the fallback and want throughput,
check whether Evo2()/StripedHyena can be re-initialised with a larger max_batch_size.
"""
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def _pool(e, pool):
    if pool == "mean": return e.mean(dim=1)
    if pool == "last": return e[:, -1]
    return e.max(dim=1).values


def _forward(evo2_model, wins, layer, pool, device):
    import torch
    ids = [evo2_model.tokenizer.tokenize(w) for w in wins]        # wins MUST be equal length
    batch = torch.tensor(ids, dtype=torch.int, device=device)     # (B, L)
    with torch.no_grad():
        _, emb = evo2_model(batch, return_embeddings=True, layer_names=[layer])
    return _pool(emb[layer].float(), pool).cpu().numpy()          # (B, D)


def embed_sequences(evo2_model, seqs, layer="blocks.28.mlp.l3", pool="mean",
                    batch_size=16, device="cuda:0", log_every=2000):
    """(N,) sequences -> (N, D) pooled Evo2 embeddings. Batches equal-length windows; on a batched-
    forward failure, falls back to per-item for the remainder (Vortex max_batch_size=1 safety)."""
    import numpy as np
    can_batch = True
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))   # group equal lengths into rectangular batches
    out = [None] * len(seqs)
    done, i = 0, 0
    while i < len(order):
        L0 = len(seqs[order[i]])
        j = i
        while j < len(order) and (j - i) < batch_size and len(seqs[order[j]]) == L0:
            j += 1
        idx = order[i:j]
        wins = [seqs[k] for k in idx]
        if can_batch and len(wins) > 1:
            try:
                vecs = _forward(evo2_model, wins, layer, pool, device)
            except Exception as e:
                can_batch = False
                print(f"  [evo2_common] batched forward failed ({type(e).__name__}: {e}); "
                      f"falling back to batch_size=1 for the rest of the run.", flush=True)
                vecs = np.stack([_forward(evo2_model, [w], layer, pool, device)[0] for w in wins])
        else:
            vecs = np.stack([_forward(evo2_model, [w], layer, pool, device)[0] for w in wins])
        for k, v in zip(idx, vecs):
            out[k] = v
        done += len(idx); i = j
        if done % log_every < len(idx):
            print(f"  embedded {done}/{len(seqs)}", flush=True)
    return np.asarray(out)
