import pandas as pd

# 1. Load isolated CDS file
input_file = "geneCodingSequences.csv" 
df = pd.read_csv(input_file)

# Get the name of sequence column
column_name = df.columns[0]

# 2. Slice the first 81 bases from each sequence
df['sliced_sequence'] = df[column_name].apply(lambda x: str(x)[:81])

# 3. Add the classification label '0' for negative controls
df['label'] = 0

# 4. Drop the original massive column so we only keep our clean ML features
final_df = df[['sliced_sequence', 'label']]

# 5. Save directly to negative dataset file
output_file = "nonprom_with_labels.csv"
final_df.to_csv(output_file, index=False)

print(f"Success! Slices complete. Saved 81-base negative controls to {output_file}")
print("\nFirst row sample:")
print(final_df.head(1))