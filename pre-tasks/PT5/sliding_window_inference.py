import Bio
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import torch 
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from prokbert.models import ProkBertForSequenceClassification

#defining the model & the FASTA file / genome that needs to be used
fasta_path = "MG1655.fna"
model_name = "neuralbioinfo/prokbert-mini-long-promoter"

#defining params
window_size = 80
step_size = 5
batch_size = 64

#model params 
kmer = 6
shift = 2

tokenizer = ProkBERTTokenizer(
    tokenization_params={"kmer":kmer, "shift": shift}
)

model = ProkBertForSequenceClassification.from_pretrained(
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
    for row in batch_rows: 
        sequences = [row["sequence"]]
        inputs = tokenizer.encode_plus(row["sequence"])

    inputs = {
        key: torch.tensor(value).unsqueeze(0).long()
    for key, value in inputs.items()
    }
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]

    for row, prob in zip(batch_rows, probs.cpu().tolist()):
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

df.to_csv("mg1655_promoter_predictions.csv", index=False)
