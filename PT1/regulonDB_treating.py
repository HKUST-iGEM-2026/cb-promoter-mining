import pandas as pd

# 1. Load your existing isolated promoter CSV file
input_file = "regulonDB_prom.csv"
df = pd.read_csv(input_file)

# 2. Add a new column named 'label' (or whatever you prefer) and set all its values to 1
# This appends a ',1' to every single row automatically
df['label'] = 1

# 3. Save it as your final training dataset file
output_file = "promoters_with_labels.csv"
df.to_csv(output_file, index=False)

print(f"Successfully added labels! Saved training-ready file to: {output_file}")