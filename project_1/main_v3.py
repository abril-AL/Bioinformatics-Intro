import numpy as np
import glob
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

# === Load batches ===

x_batches = sorted(glob.glob("train_X_*.npy"))
y_batches = sorted(glob.glob("train_y_*.npy"))
total_batches = len(x_batches)

# === Load validation batch ===
val_X = np.load(x_batches[-1])
val_y_raw = np.load(y_batches[-1])

label_encoder = LabelEncoder()
val_y_encoded = label_encoder.fit_transform(val_y_raw)

# === Standardize validation features ===
scaler = StandardScaler()
val_X_scaled = scaler.fit_transform(val_X)

# === Define SGDClassifier ===
clf = SGDClassifier(
    loss='log_loss',
    alpha=1e-3, # generalize better ?
    penalty='l2',
    learning_rate='optimal',
    max_iter=5,
    warm_start=True,
    n_jobs=-1
)

classes = np.unique(val_y_encoded)

# === Multi-epoch training ===
for epoch in range(10):
    for i in range(total_batches - 1):  # exclude validation batch
        X = np.load(x_batches[i])
        y = np.load(y_batches[i])
        y_encoded = label_encoder.transform(y)

        X_scaled = scaler.fit_transform(X)

        if epoch == 0 and i == 0:
            clf.partial_fit(X_scaled, y_encoded, classes=classes)
        else:
            clf.partial_fit(X_scaled, y_encoded)
    val_X_scaled = scaler.transform(val_X)
    y_pred = clf.predict(val_X_scaled)
    acc = (y_pred == val_y_encoded).mean()
    print(f"Epoch {epoch+1} accuracy: {acc:.4f}")

# === Predict on validation set ===
def batched_predict(clf, X_val, batch_size=100000):
    preds = []
    for i in tqdm(range(0, len(X_val), batch_size), desc="Predicting in batches"):
        batch = X_val[i:i + batch_size]
        batch_preds = clf.predict(batch)
        preds.append(batch_preds)
    return np.concatenate(preds)

y_pred = batched_predict(clf, val_X_scaled)

print("\n=== Validation Performance ===")
print(classification_report(val_y_encoded, y_pred, target_names=label_encoder.classes_))