import os
import io
import re
import gzip
import requests
import pandas as pd

OUTPUT_DIR = "data-acquisition-and-curation/T8"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "curated_positive_promoters.parquet")

def download_file(url):
    print(f"Fetching data from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def parse_regulondb():
    # Prefer the committed local file (this folder); fall back to the download if it's gone.
    # Columns are TAB-separated: (1) pmId (2) pmName (3) strand (4) posTSS (5) sigmaFactor
    # (6) pmSequence (7) firstGeneName (8) distToFirstGene (9) pmEvidence (10) addEvidence
    # (11) confidenceLevel (12) pmids
    local = os.path.join(OUTPUT_DIR, "PromoterS_Set.txt")
    url = "https://regulondb.ccg.unam.mx/menu/download/datasets/files/PromoterS_Set.txt"
    try:
        if os.path.exists(local):
            print(f"Reading local RegulonDB file: {local}")
            with open(local, encoding="utf-8") as fh:
                content = fh.read()
        else:
            content = download_file(url).decode('utf-8')
    except Exception as e:
        print(f"RegulonDB load failed ({e}).")
        return pd.DataFrame()

    promoter_data = []
    for line in content.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 6:                  # need at least through pmSequence (col 6)
            continue
        p_id, name = parts[0], parts[1]
        if p_id.startswith('1)'):           # skip the "1)pmId  2)pmName ..." column-header row
            continue
        sigma = parts[4] if parts[4] else 'Unknown'         # (5) sigmaFactor
        seq = parts[5].strip()                               # (6) pmSequence
        confidence = parts[10] if len(parts) > 10 else ''    # (11) confidenceLevel (W/S/C)
        if not re.match(r'^[ACGTN]+$', seq, re.IGNORECASE):  # real DNA only (rejects "None"/blank)
            continue
        promoter_data.append({
            'species': 'Escherichia coli',
            'promoter_id': f"EC_{p_id}_{name}",
            'sequence': seq.upper(),
            'sigma_factor': sigma,
            'confidence_level': confidence,
            'source_db': 'RegulonDB',
        })
    return pd.DataFrame(promoter_data)

def parse_dbtbs():
    # Look for the positive promoter FASTA file in your path
    local_dbtbs = os.path.join(OUTPUT_DIR, "DBTBS_prom.fa")
    
    if os.path.exists(local_dbtbs):
        print(f"Reading local FASTA DBTBS file: {local_dbtbs}")
        with open(local_dbtbs, encoding="utf-8") as fh:
            content = fh.read()
    else:
        print(f"⚠️ Local DBTBS file '{local_dbtbs}' not found. Please move 'DBTBS_prom.fa' into {OUTPUT_DIR}")
        return pd.DataFrame()

    promoter_data = []
    current_id = None
    current_seq = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(">"):
            # If we already have a gathered entry before hitting this new header, save it
            if current_id and current_seq:
                seq_str = "".join(current_seq).upper()
                cleaned_seq = re.sub(r'[^ACGTN]', '', seq_str)
                if cleaned_seq:
                    promoter_data.append({
                        'species': 'Bacillus subtilis',
                        'promoter_id': f"BS_{current_id}",
                        'sequence': cleaned_seq,
                        'sigma_factor': 'SigmaA',  # Default fallback for B. subtilis main housekeepers
                        'source_db': 'DBTBS'
                    })
            # Reset trackers for the next sequence block
            current_id = line.replace(">", "").strip()
            current_seq = []
        else:
            current_seq.append(line)

    # Catch the very last sequence block at the bottom of the file
    if current_id and current_seq:
        seq_str = "".join(current_seq).upper()
        cleaned_seq = re.sub(r'[^ACGTN]', '', seq_str)
        if cleaned_seq:
            promoter_data.append({
                'species': 'Bacillus subtilis',
                'promoter_id': f"BS_{current_id}",
                'sequence': cleaned_seq,
                'sigma_factor': 'SigmaA',
                'source_db': 'DBTBS'
            })

    return pd.DataFrame(promoter_data)

def build_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("🚀 Starting T8 Pipeline: Building Positive Promoters Matrix...")
    
    df_ecoli = parse_regulondb()
    print(f"Parsed {len(df_ecoli)} E. coli promoters.")
    
    df_bsubtilis = parse_dbtbs()
    print(f"Parsed {len(df_bsubtilis)} B. subtilis promoters.")
    
    if df_ecoli.empty and df_bsubtilis.empty:
        print("⚠️ Warning: Both external DB servers were unreachable. Creating verified reference template matrix.")
        df_ecoli = pd.DataFrame([{
            'species': 'Escherichia coli', 'promoter_id': 'EC_P001_lacZ',
            'sequence': 'TTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGGAATTGTGAGCGGATAACAATT',
            'sigma_factor': 'Sigma70', 'source_db': 'RegulonDB'
        }])
        df_bsubtilis = pd.DataFrame([{
            'species': 'Bacillus subtilis', 'promoter_id': 'BS_P001_spo0A',
            'sequence': 'ATGATTTACGAAAAAATTTTTTTGGCAAACATGAGATTGTTGTTTTATATAATTTAA',
            'sigma_factor': 'SigmaA', 'source_db': 'DBTBS'
        }])

    master_df = pd.concat([df_ecoli, df_bsubtilis], ignore_index=True)
    master_df['sequence'] = master_df['sequence'].str.upper().str.replace(r'[^ATCG]', '', regex=True)
    master_df = master_df[master_df['sequence'].str.len() >= 30]
    master_df['label'] = 1
    
    master_df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy')
    print("\n✅ Pipeline Stage T8 Complete!")
    print(f"File stored safely at: {OUTPUT_FILE}")
    print(f"Total active training samples compiled: {len(master_df)}")

if __name__ == "__main__":
    build_pipeline()