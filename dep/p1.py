import pandas as pd
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm
# mango fruit tea brown sugar boba 25% sweet 

# Step 1: Load FASTA sequences
def load_sequences(fasta_path):
    print("> Loading Sequences from FASTA...")
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}
    print(f"> Loaded {len(sequences)} sequences.")
    example_id = next(iter(sequences))
    print(f"> Example: {example_id} → {sequences[example_id][:20]}... ({len(sequences[example_id])} aa)")
    return sequences

# Step 2: Load training and test data
def load_labels(tsv_path):
    print("> Loading Labels from TSV...")
    df = pd.read_csv(tsv_path, sep='\t')
    df[['pdb_id', 'aa', 'pos']] = df['id'].str.extract(r'(\w+)_(\w+)_(\d+)')
    print(f"> Loaded {len(df)} label entries.")
    print(f"> Example row:\n{df.head(1)}")
    return df

# Step 3: Get embeddings from ESM
class ESMEmbedder:
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        print("> Loading ESM model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print("> Model ready.")

    def embed_sequence(self, sequence):
        #print(f"> Embedding sequence of length {len(sequence)}")
        tokens = self.tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**tokens)
        embedding = output.last_hidden_state[0, 1:-1].mean(dim=0).numpy()
        #print(f"> Embedding shape: {embedding.shape}")
        return embedding

# Step 4: Train simple classifier
def train_classifier(X, y):
    print("> Scaling and training classifier...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_scaled, y)
    print(f"> Training complete. Classes: {clf.classes_}")
    return clf, scaler

# Step 5: Predict and evaluate
def evaluate(clf, X_test, y_test):
    print("> Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"> Accuracy: {acc:.4f}")
    return acc

# Orchestrate
def main():
    sequences = load_sequences("sequences.fasta")
    train_df = load_labels("train.tsv")
    
    embedder = ESMEmbedder()

    # Embed each sequence once
    print("> Precomputing embeddings...")
    protein_embeddings = {}
    for pdb_id, seq in tqdm(sequences.items(), desc="> Embedding proteins"):
        try:
            protein_embeddings[pdb_id] = embedder.embed_sequence(seq)
        except Exception as e:
            print(f"!! Failed to embed {pdb_id}: {e}")

    # Build training matrix using cached embeddings
    X_train, y_train = [], []
    print("> Building feature matrix...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="> Building feature matrix"):
        pdb_id = row['pdb_id']
        if pdb_id not in protein_embeddings:
            continue
        embedding = protein_embeddings[pdb_id]
        X_train.append(embedding)
        y_train.append(row['secondary_structure'])

        if idx < 3:
            print(f"> ID: {row['id']} → Label: {row['secondary_structure']}")
            print(f"> First 5 dims: {embedding[:5]}")

    print(f"> Final feature matrix shape: {np.array(X_train).shape}")
    clf = train_classifier(X_train, y_train)

    # Evaluate
    from sklearn.model_selection import train_test_split
    X, y = np.array(X_train), np.array(y_train)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    clf, scaler = train_classifier(X_train, y_train)

    X_val_scaled = scaler.transform(X_val)
    evaluate(clf, X_val_scaled, y_val)

if __name__ == "__main__":
    main()