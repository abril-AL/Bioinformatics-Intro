import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import glob
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
from Bio import SeqIO
import batching
import pickle

print("🔄 Loading training dataframe...")
train_df = pd.read_csv("project_1/train.tsv", sep='\t')
id_parts = train_df['id'].str.split('_', expand=True)
train_df['protein_id'] = id_parts[0]
train_df['residue_index'] = id_parts[2].astype(int)

print("🔄 Loading and parsing FASTA sequences...")
sequence_dict = {}
for record in tqdm(SeqIO.parse("project_1/sequences.fasta", "fasta"), desc="Parsing sequences"):
    sequence_dict[record.id.strip()] = str(record.seq)

print("🔄 Loading ESM model...")
model_name = "facebook/esm2_t12_35M_UR50D"  # or 650M
#model_name = "facebook/esm2_t6_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embedding_cache_file = "esm_embeddings.pkl"
if os.path.exists(embedding_cache_file):
    print("📦 Loading cached embeddings from disk...")
    with open(embedding_cache_file, "rb") as f:
        embedding_dict = pickle.load(f)
else:
    print("🚀 Generating embeddings...")
    embedding_dict = {}
    for protein_id, sequence in tqdm(sequence_dict.items(), desc="Embedding proteins"):
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0, 1:-1, :]
        for i, emb in enumerate(token_embeddings):
            embedding_dict[(protein_id, i + 1)] = emb.cpu().numpy()[:50]  # truncate for memory
    with open(embedding_cache_file, "wb") as f:
        pickle.dump(embedding_dict, f)
    print("✅ Saved embeddings to disk.")

print("🔄 Processing and saving training batches...")
batching.process_and_save_batches(train_df, embedding_dict, batch_size=25000)

x_batches = sorted(glob.glob("train_X_*.npy"))
y_batches = sorted(glob.glob("train_y_*.npy"))
total_batches = len(x_batches)

val_X = np.load(x_batches[-1])
val_y = np.load(y_batches[-1])

label_encoder = LabelEncoder()
val_y_encoded = label_encoder.fit_transform(val_y)

print("Starting batch-wise training...")
clf = SGDClassifier(
    loss='log_loss',
    alpha=1e-3,
    penalty='l2',
    learning_rate='optimal',
    max_iter=1,
    warm_start=True,
    n_jobs=-1
)

classes = np.unique(val_y_encoded)

for epoch in range(15):
    print(f"\nEpoch {epoch+1}/15")
    for i in range(total_batches - 1):
        #print(f"\tTraining on batch {i+1}/{total_batches - 1}")
        X = np.load(x_batches[i])
        y = np.load(y_batches[i])
        y_encoded = label_encoder.transform(y)
        if epoch == 0 and i == 0:
            clf.partial_fit(X, y_encoded, classes=classes)
        else:
            clf.partial_fit(X, y_encoded)
    y_pred = clf.predict(val_X)
    acc = (y_pred == val_y_encoded).mean()
    print(f"Epoch {epoch+1} accuracy: {acc:.4f}")

print("\nFinal evaluation on validation set...")
def batched_predict(clf, X_val, batch_size=100000):
    preds = []
    for i in tqdm(range(0, len(X_val), batch_size), desc="Predicting in batches"):
        batch = X_val[i:i + batch_size]
        batch_preds = clf.predict(batch)
        preds.append(batch_preds)
    return np.concatenate(preds)

y_pred = batched_predict(clf, val_X)
print("\n=== Validation Performance ===")
print(classification_report(val_y_encoded, y_pred, target_names=label_encoder.classes_))