import os
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA, IncrementalPCA


# Load model and tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Make cache directory
os.makedirs("embeddings", exist_ok=True)

# Load sequences
fasta_path = "project_1/sequences.fasta"
sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}

# Batch embedding function
@torch.no_grad()
def get_batch_embeddings(batch_pids, batch_seqs):
    tokens = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True).to(device)
    outputs = model(**tokens)
    all_embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']

    for i, pid in enumerate(batch_pids):
        # Find valid tokens (excluding [CLS] and [EOS])
        length = attention_mask[i].sum().item() - 2
        emb = all_embeddings[i, 1:1+length, :].cpu().numpy()
        with open(f"embeddings/{pid}.pkl", "wb") as f:
            pickle.dump(emb, f)

# Load training data
train_df = pd.read_csv("project_1/train.tsv", sep="\t")
train_df[['protein_id', 'residue', 'position']] = train_df['id'].str.extract(r"(\w+)_(\w+)_(\d+)")
train_df['position'] = train_df['position'].astype(int) - 1

# Generate and cache embeddings
protein_embeddings = {}
batch_size = 16
batch_pids, batch_seqs = [], []

for pid, seq in tqdm(sequences.items()):
    cache_file = f"embeddings/{pid}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            protein_embeddings[pid] = pickle.load(f)
    else:
        if len(seq) < 1:
            print(f"Skipping {pid} — empty sequence")
            continue
        batch_pids.append(pid)
        batch_seqs.append(seq)
        if len(batch_pids) == batch_size:
            try:
                get_batch_embeddings(batch_pids, batch_seqs)
            except Exception as e:
                print(f"Error processing batch {batch_pids}: {e}")
            batch_pids, batch_seqs = [], []

# Final batch
if batch_pids:
    try:
        get_batch_embeddings(batch_pids, batch_seqs)
    except Exception as e:
        print(f"Error processing batch {batch_pids}: {e}")

# Reload all cached embeddings
for pid in sequences:
    cache_file = f"embeddings/{pid}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            protein_embeddings[pid] = pickle.load(f)

# Build training matrix
X, y = [], []
label_map = {
    'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'P': 5, 'T': 6, 'S': 7, '.': 8
}

for _, row in train_df.iterrows():
    pid = row['protein_id']
    pos = row['position']
    label = row['secondary_structure']
    if pid in protein_embeddings and pos < len(protein_embeddings[pid]):
        X.append(protein_embeddings[pid][pos])
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)
# subsample for mem management 
X = X[:100000]
y = y[:100000]

# Apply (incrremental) PCA to reduce dimensionality
ipca = IncrementalPCA(n_components=100, batch_size=10000)
X_reduced = ipca.fit_transform(X)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_reduced, y)

# Load and prepare test data
test_df = pd.read_csv("project_1/test.tsv", sep="\t")
test_df[['protein_id', 'residue', 'position']] = test_df['id'].str.extract(r"(\w+)_\w+_(\d+)")
test_df['position'] = test_df['position'].astype(int) - 1

# Make predictions
inverse_label_map = {v: k for k, v in label_map.items()}
X_test = []
valid_indices = []

for i, row in test_df.iterrows():
    pid = row['protein_id']
    pos = row['position']
    if pid in protein_embeddings and pos < len(protein_embeddings[pid]):
        X_test.append(protein_embeddings[pid][pos])
        valid_indices.append(i)

X_test = np.array(X_test)
X_test_reduced = pca.transform(X_test)
y_pred = clf.predict(X_test_reduced)

# Map predictions back to labels
pred_labels = [inverse_label_map[i] for i in y_pred]
test_df['prediction'] = '.'
for idx, label in zip(valid_indices, pred_labels):
    test_df.at[idx, 'prediction'] = label

# Save output
test_df[['id', 'prediction']].to_csv("prediction.csv", index=False)
# zip prediction.zip prediction.csv
