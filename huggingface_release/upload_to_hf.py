#!/usr/bin/env python3
"""
upload_to_hf.py — publish the three HKUST iGEM 2026 deployment models to the Hugging Face Hub,
each with its model card (the sibling <repo>/README.md).

  prokbert-mini-clostridium-promoter        ProkBERT-mini fine-tune (HPC4 checkpoint dir)   CC-BY-NC-4.0
  cnn-fusion-clostridium-promoter           multi-encoding fusion CNN (HPC4 checkpoint dir) CC-BY-4.0
  evo2-logreg-clostridium-promoter-probe    logreg probe on Evo2-7B embeddings (in-repo)    CC-BY-4.0

  ⚠ RECONSTRUCTED 2026-09-11 to match huggingface_release/README.md. The original was never committed
  to git and was accidentally overwritten; this rebuild follows the documented interface (HF_OWNER +
  two HPC4 checkpoint paths below, --only / --dry-run / --public). Sanity-check it before you push.

No token lives in this file. It uses the token cached by `huggingface-cli login` (your step). Repos are
created PRIVATE; review on the Hub, then make public in the UI or re-run with --public.

  pip install -U huggingface_hub
  huggingface-cli login                 # WRITE token
  #   edit HF_OWNER + the two HPC4 paths below
  python upload_to_hf.py --only evo2    # evo2 needs no HPC4 paths — its files are in-repo
  python upload_to_hf.py --dry-run      # check what each repo would receive
  python upload_to_hf.py                # push all three (private)
"""
# ================= EDIT THESE =================
HF_OWNER = "CloBiT"      # your HF org or username
PROKBERT_CKPT = "/Users/oak23/cb-ckpts/prokbert_best_model"   # ProkBERT fine-tune checkpoint DIR (config.json + *.safetensors + tokenizer)
CNN_CKPT = "/Users/oak23/cb-ckpts/cnn_v2"                     # CNN-v2 checkpoint DIR (best_model.pt + config.json)
# =============================================

import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
EVO2_DIR = os.path.join(REPO_ROOT, "model-training", "Evo2")          # probe + helper live here (in-repo)
CNN_CODE = os.path.join(REPO_ROOT, "model-training", "cnn", "train_cnn_v2.py")

# name -> (hub repo == card dir, kind, source spec, extra single files to add)
MODELS = {
    "prokbert": dict(repo="prokbert-mini-clostridium-promoter", kind="dir",
                     src=lambda: PROKBERT_CKPT, extra=[]),
    "cnn":      dict(repo="cnn-fusion-clostridium-promoter", kind="dir",
                     src=lambda: CNN_CKPT, extra=[CNN_CODE]),          # + train_cnn_v2.py
    "evo2":     dict(repo="evo2-logreg-clostridium-promoter-probe", kind="files",
                     files=[os.path.join(EVO2_DIR, "evo2_promoter_classifier.pkl"),
                            os.path.join(EVO2_DIR, "evo2_common.py"),
                            os.path.join(HERE, "evo2-logreg-clostridium-promoter-probe", "apply_probe.py")]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(MODELS), help="publish just one model (default: all three)")
    ap.add_argument("--public", action="store_true", help="create PUBLIC repos (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="check inputs + auth, upload nothing")
    a = ap.parse_args()

    if not HF_OWNER:
        sys.exit("Edit HF_OWNER at the top of this file first (your HF org or username).")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed:  pip install -U huggingface_hub")

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run `huggingface-cli login` with a WRITE token first (your step).")
    print(f"auth: {who}   owner: {HF_OWNER}   visibility: {'PUBLIC' if a.public else 'private'}")

    names = [a.only] if a.only else list(MODELS)

    # preflight: every source path + card must exist before any repo is touched
    problems = []
    for n in names:
        m = MODELS[n]
        card = os.path.join(HERE, m["repo"], "README.md")
        if not os.path.isfile(card):
            problems.append(f"[{n}] missing card {card}")
        if m["kind"] == "dir":
            if not m["src"]() or not os.path.isdir(m["src"]()):
                problems.append(f"[{n}] checkpoint dir not set/found (edit the path at top): {m['src']()!r}")
            for e in m["extra"]:
                if not os.path.isfile(e):
                    problems.append(f"[{n}] missing extra file {e}")
        else:
            for f in m["files"]:
                if not os.path.isfile(f):
                    problems.append(f"[{n}] missing file {f}")
    if problems:
        sys.exit("Preflight failed:\n  " + "\n  ".join(problems))

    for n in names:
        m = MODELS[n]
        repo_id = f"{HF_OWNER}/{m['repo']}"
        print(f"\n=== {repo_id} ===")
        if a.dry_run:
            src = m["src"]() if m["kind"] == "dir" else ""
            items = ([f"folder:{src}"] + m["extra"]) if m["kind"] == "dir" else m["files"]
            print("  would upload:", *[os.path.basename(x) or x for x in items], "+ README.md")
            continue

        api.create_repo(repo_id, repo_type="model", private=not a.public, exist_ok=True)
        if m["kind"] == "dir":
            api.upload_folder(repo_id=repo_id, folder_path=m["src"](), repo_type="model",
                              ignore_patterns=["*.tmp", "__pycache__/*", "optimizer.pt", "checkpoint-*/*"])
            for e in m["extra"]:
                api.upload_file(repo_id=repo_id, path_or_fileobj=e,
                                path_in_repo=os.path.basename(e), repo_type="model")
        else:
            for f in m["files"]:
                api.upload_file(repo_id=repo_id, path_or_fileobj=f,
                                path_in_repo=os.path.basename(f), repo_type="model")

        # upload the card with the <HF_OWNER> placeholder resolved
        card_txt = open(os.path.join(HERE, m["repo"], "README.md")).read().replace("<HF_OWNER>", HF_OWNER)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(card_txt); tmp = fh.name
        api.upload_file(repo_id=repo_id, path_or_fileobj=tmp, path_in_repo="README.md", repo_type="model")
        os.unlink(tmp)
        print(f"  done -> https://huggingface.co/{repo_id}")

    if not a.dry_run:
        print(f"\nComplete. Repos are {'PUBLIC' if a.public else 'PRIVATE — review, then flip public in the HF UI'}.")


if __name__ == "__main__":
    main()
