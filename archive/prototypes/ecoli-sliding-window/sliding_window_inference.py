from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#defining the model & the FASTA file / genome that needs to be used
fasta_path = "MG1655.fna"
model_name = "neuralbioinfo/prokbert-mini-promoter"

#defining params
window_size = 81      # matches the model's training window (RegulonDB promoter = 81 bp, +1 at pos 61)
step_size = 5
batch_size = 64

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

record = next(SeqIO.parse(fasta_path,"fasta"))

genome = str(record.seq).upper()
chromosome = record.id
results = []

def predict_batch(batch_rows):
    # tokenize the WHOLE batch and score it in one forward pass.
    # every window is the same length, so the token sequences line up with no padding.
    inputs = tokenizer([r["sequence"] for r in batch_rows], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()

    for row, prob in zip(batch_rows, probs):
        row["promoter_score"] = prob
        results.append(row)

def scan_sequence(seq, strand):
    batch = []

    for start in range(0, len(seq) - window_size + 1, step_size):
        end = start + window_size
        window_seq = seq[start:end]

        if strand == "+":
            orig_start = start
            orig_end = end
        else:
            orig_start = len(genome) - end
            orig_end = len(genome) - start

        batch.append({
            "chromosome": chromosome,
            "start": orig_start,
            "end": orig_end,
            "strand": strand,
            "sequence": window_seq
        })

        if len(batch) == batch_size:
            predict_batch(batch)
            batch = []

    if batch:
        predict_batch(batch)

# scans the forward genome / forward strand
print("Scanning forward strand")
scan_sequence(genome, "+")

#scans the reverse strand
print("Scanning reverse strand")
rev_genome = str(Seq(genome).reverse_complement())
scan_sequence(rev_genome, "-")

df = pd.DataFrame(results)
df = df.sort_values("promoter_score", ascending=False)

df.to_csv("mg1655_promoter_scores1.csv", index=False)
