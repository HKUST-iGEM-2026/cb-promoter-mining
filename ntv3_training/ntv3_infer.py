#!/usr/bin/env python3
"""
ntv3_infer.py — reload a trained NTv3-ctx classifier and score sequences, EXACTLY
matching finetune_ntv3_classifier.py (backbone + mask-aware mean pool + linear head,
tokenised add_special_tokens=False / pad|truncate to meta['max_length']).

Shared by ntv3_rescore_clean.py and ntv3_genome_scan.py so scoring can never drift
between the clean-rescore and the genome scan.

  from ntv3_infer import load_ntv3, score_seqs
  tok, backbone, head, meta, device = load_ntv3("<...>/final_model")
  prob1, logit = score_seqs(list_of_strings, tok, backbone, head, meta, device)
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_ntv3(model_dir, device=None):
    """Reload backbone/ + classifier_head.pt + meta.json into (tok, backbone, head, meta, device)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    bb = os.path.join(model_dir, "backbone")
    tok = AutoTokenizer.from_pretrained(bb, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(bb, trust_remote_code=True).to(device).eval()
    head = nn.Linear(meta["embed_dim"], meta["num_labels"])
    head.load_state_dict(torch.load(os.path.join(model_dir, "classifier_head.pt"),
                                    map_location=device))
    head = head.to(device).eval()
    return tok, backbone, head, meta, device


@torch.no_grad()
def score_seqs(seqs, tok, backbone, head, meta, device, batch_size=16):
    """(prob1, logit) for a list of sequences. prob1 = softmax(logits)[:,1];
    logit = logits[:,1]-logits[:,0] (the feature the T26 calibrators use). bf16 autocast
    on CUDA to match the H800 training dtype; the head runs in fp32."""
    ML = meta["max_length"]
    on_cuda = str(device).startswith("cuda")
    probs, logits_diff = [], []
    for i in range(0, len(seqs), batch_size):
        chunk = [str(s).upper() for s in seqs[i:i + batch_size]]
        enc = tok(chunk, add_special_tokens=False, padding="max_length",
                  max_length=ML, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        am = am.to(device) if am is not None else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=on_cuda):
            out = backbone(input_ids=input_ids, attention_mask=am, output_hidden_states=True)
            h = out.hidden_states[-1]
            if am is not None:
                m = am.unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            else:
                pooled = h.mean(1)
        lg = head(pooled.float())                       # fp32 head, outside autocast math
        probs.append(torch.softmax(lg, dim=-1)[:, 1].cpu().numpy())
        logits_diff.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    return np.concatenate(probs), np.concatenate(logits_diff)
