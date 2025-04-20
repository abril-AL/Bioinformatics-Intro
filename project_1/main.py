# Project 1 - CS C121
# Secondary Structure Prediction
DB = True # Debug

# -- STEP 1: Data Ingestion --
# train.tsv: Contains protein IDs, amino acid, and secondary structure labels.
# test.tsv: Contains protein IDs and amino acids only.
# sequences.fasta: Contains full sequences of each protein.
import pandas as pd
from Bio import SeqIO

train_df = pd.read_csv("train.tsv", sep='\t')
test_df = pd.read_csv("test.tsv", sep='\t')

seq_dict = {}
for record in SeqIO.parse("sequences.fasta", "fasta"):
    protein_id = record.id.strip() 
    sequence = str(record.seq)
    seq_dict[protein_id] = sequence

if DB:
    print("Train samples:", len(train_df))
    print("Test samples:", len(test_df))
    print("Loaded sequences:", len(seq_dict))
    print("Sample:", list(seq_dict.items())[0])

def parse_id(entry):
    # e.g., "3KVH_LYS_6" → ("3KVH", 6)
    parts = entry.split('_')
    return parts[0], int(parts[2])

for df in [train_df, test_df]: # Add id and pos to dataframe
    df[['protein_id', 'residue_index']] = df['id'].apply(
        lambda x: pd.Series(parse_id(x))
    )

# --- STEP 2: Feature Extraction ---
from transformers import AutoTokenizer, AutoModel
import torch

# load esm model
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate ESM embeddings - tokenize protein sequence
# seq_dict[(protein_id, position)] = embedding_vector


# --- STEP 3: Split Data Set ---


# --- STEP 4: Train Classifier --- 


# --- STEP 5: Made Predictions --- 


# --- STEP 6: Export or Codabench Submission ---