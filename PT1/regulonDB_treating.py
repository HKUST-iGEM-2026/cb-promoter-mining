import pandas as pd

# 1. Load the TSV file
# We use sep='\t' because TSV files are tab-separated
input_file = "Gene_sequence.tsv"  # Replace with your actual file name
df = pd.read_csv(input_file, sep='\t')

# 2. Inspect your columns to find the exact name of the promoter part
print("Columns found in your file:")
print(df.columns.tolist())

# 3. Specify the exact column name containing the promoter sequences
# Replace 'promoter_sequence' with the exact header name printed above
# promoter_column_name = '10)sequence' 

# If your file does not have a header row, you can use the column index instead (0-indexed).
promoter_column_name = df.columns[9]

# 4. Filter the dataframe to keep ONLY that single column
cleaned_df = df[[promoter_column_name]]

# 5. Export directly to a clean CSV format
output_file = "geneCodingSequences.csv"
cleaned_df.to_csv(output_file, index=False)

print(f"\nSuccess! Kept only column '{promoter_column_name}' and saved to {output_file}")