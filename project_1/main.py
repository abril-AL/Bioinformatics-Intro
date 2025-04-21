# Project 1 - CS C121
# Secondary Structure Prediction
DB = True # Debug
# conda install -c conda-forge pandas biopython scikit-learn tqdm -y
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
# pip install transformers
# python -c "import torch; import transformers; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow info/warning/error logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # HuggingFace warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# -- STEP 1: Data Ingestion --
# train.tsv: Contains protein IDs, amino acid, and secondary structure labels.
# test.tsv: Contains protein IDs and amino acids only.
# sequences.fasta: Contains full sequences of each protein.
import pandas as pd
from Bio import SeqIO


train_df = pd.read_csv("project_1/train.tsv", sep='\t')
test_df = pd.read_csv("project_1/test.tsv", sep='\t')

seq_dict = {}
for record in SeqIO.parse("project_1/sequences.fasta", "fasta"):
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

for df in [train_df, test_df]:
    id_parts = df['id'].str.split('_', expand=True)
    df['protein_id'] = id_parts[0]
    df['residue_index'] = id_parts[2].astype(int)

# --- STEP 2: Feature Extraction ---
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import torch
import time

# load esm model
print("Loading ESM model")
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# try to use GPU
print("Checking for GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate ESM embeddings - tokenize protein sequence
embedding_cache_file = "esm_embeddings.pkl"
if os.path.exists(embedding_cache_file):
    # cache so i dont kill my laptop
    with open(embedding_cache_file, "rb") as f:
        embedding_dict = pickle.load(f)
    print("Loaded cached embeddings from file.")
else:
    print("Embedding proteins...")
    embedding_dict = {}
    for protein_id, sequence in tqdm(seq_dict.items(), desc="Embedding proteins", unit="seq"):
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Shape: (seq_len, hidden_dim)
        token_embeddings = outputs.last_hidden_state[0, 1:-1, :]  # slice off start/end tokens

        for i, emb in enumerate(token_embeddings):
            embedding_dict[(protein_id, i + 1)] = emb.cpu().numpy()  # 1-based indexing

    # Save embeddings to disk
    with open(embedding_cache_file, "wb") as f:
        pickle.dump(embedding_dict, f)
    print("Saved embeddings to disk.")

# --- STEP 3: Split Data Set ---

# --- STEP 4: Train Classifier --- 


# --- STEP 5: Made Predictions --- 


# --- STEP 6: Export or Codabench Submission ---