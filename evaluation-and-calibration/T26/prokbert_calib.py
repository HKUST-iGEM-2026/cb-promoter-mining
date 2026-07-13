from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#practice with the un-tuned model 
model_name = "neuralbioinfo/prokbert-mini-promoter"
model_path = "/project/igem2026/cb-promoter-mining/results/T21/prokbert/trial_12/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
model.eval()

data = pd.read_csv("ecoli_calib_set.csv")
# takes the sequences as an ordered series
promoters = data["Sequence"]    
scores = []

# takes index in batches, so i goes from 0, 32, 64
# window size is 81, batch size is 32
for i in range(0, len(promoters), 32): 
    batch = promoters[i:i+32].tolist()
    inputs = tokenizer(batch, padding=max_length, max_length=81, return_tensors="pt")

    with torch.no_grad(): 
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:,1].tolist()

    scores.extend(probs)

data["Score"] = scores
print(tokenizer(["A" * 52], padding="max_length", max_length=78, return_tensors="pt").keys()) 

# no need to train prokBERT we just need to have some test data & then the calibrator data & see if the prediction improves
# will do simple logistic regression model
# x has to be 2D for logreg

x,y = data[["Score"]],data["Target"]

# having the test group be 30%, and the rest be calibration, just to check since we're not changing anything

x_calib, x_test, y_calib, y_test = train_test_split(x,y,test_size=0.3)
clf = LogisticRegression()
clf.fit(x_calib, y_calib)

# takes the x_test and turns it into a dataframe, this is the test set that shows actual success or not after the calibrator is fitted onto the data

calibrated_probs = clf.predict_proba(x_test)[:, 1]

# calib probs are in same order as y_test & x_test because both of them were cut by train_test_split tgt
# calibrated probs are in a numpy array so we turn everything else so there's no weird mismatching
# ignore pandas labels and just use everything in their current order
 
df=pd.DataFrame(
    {"Target": y_test.reset_index(drop=True),
    "ProkBERTScore": x_test["Score"].reset_index(drop=True),
     "CalibScore": calibrated_probs,
     }
)

df.to_csv("finalized_data_prokbert.csv", index=False)
