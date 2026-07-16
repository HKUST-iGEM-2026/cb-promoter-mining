from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model_name = "neuralbioinfo/prokbert-mini-promoter"
model_path = "/project/igem2026/cb-promoter-mining/results/T21/prokbert/trial_12/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
model.to(device)
model.eval()

data = pd.read_csv("/project/igem2026/cb-promoter-mining/data-acquisition-and-curation/T9/literature_promoters.tsv", sep="\t")


def at_fraction(seq):
    seq = seq.upper()
    return (seq.count("A") + seq.count("T")) / len(seq)

data["AT"] = data["Sequence"].apply(at_fraction)

# takes the sequences as an ordered series
promoters = data["Sequence"]
probs = []    # raw model probability, kept for reporting
logits = []   # positive-class log-odds, the calibrator feature

# takes index in batches, so i goes from 0, 32, 64
# window size is 81, batch size is 32
for i in range(0, len(promoters), 32):
    batch = promoters[i:i+32].tolist()
    inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=81, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # log-odds of the positive class = logit for Platt/logistic calibration.
        # softmax(...)[:,1] makes it into probabilities from 0 to 1 so it's not the full picture so shouldn't be fed into the calibrator 
        logits.extend((outputs.logits[:, 1] - outputs.logits[:, 0]).tolist())
        probs.extend(torch.softmax(outputs.logits, dim=1)[:, 1].tolist())

data["Score"] = probs     # raw probabilities from the model w/o calibrator for the before/after comparison
data["Logit"] = logits    # what's actually fed goes into the calibrator

# will do simple logistic regression model. the calibrator is fit on the logit difference which is platt scaling (but 2D) technically so calibrated_prob = s(a*logit + b*AT + c)

# split rows so we keep both the raw score (for reporting) and the logit (calibrator feature) aligned

calib_df, test_df = train_test_split(
    data, test_size=0.3, random_state=42, stratify=data["Target"]
)

features = ["Logit", "AT"] #stores the column names
clf = LogisticRegression(penalty=None)
clf.fit(calib_df[features], calib_df["Target"])
calibrated_probs = clf.predict_proba(test_df[features])[:, 1]
  # x has to be 2D for logreg so use double brackets, since "features" & also has 2 input variables

calibrated_probs = clf.predict_proba(test_df[["Logit"]])[:, 1]

# use .values throughout so pandas indices can't mismatch after the split

df = pd.DataFrame(
    {
        "Target": test_df["Target"].values,
        "ProkBERTScore": test_df["Score"].values,   # raw probability uncalibrated for "before" comparison
        "CalibScore": calibrated_probs,             # after logistic calibration so should be more accurate
    }
)

df.to_csv("finalized_data_prokbert.csv", index=False)
