import xgboost as xgb
import numpy as np
import glob
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# === Load batches ===
x_batches = sorted(glob.glob("train_X_*.npy"))
y_batches = sorted(glob.glob("train_y_*.npy"))
total_batches = len(x_batches)

# === Fit label encoder on validation batch ===
label_encoder = LabelEncoder()
val_y_raw = np.load(y_batches[-1])
label_encoder.fit(val_y_raw)
val_y = label_encoder.transform(val_y_raw)
val_X = np.load(x_batches[-1])
dval = xgb.DMatrix(val_X, label=val_y)

# === XGBoost training parameters ===
params = {
    "objective": "multi:softprob",
    "num_class": len(label_encoder.classes_),
    "max_depth": 6,
    "eta": 0.1,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "nthread": -1
}

booster = None

# === Train on all batches except validation ===
for i in range(total_batches - 1):
    print(f"Training on batch {i+1}/{total_batches - 1}")
    X = np.load(x_batches[i])
    y = label_encoder.transform(np.load(y_batches[i]))
    dtrain = xgb.DMatrix(X, label=y)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dval, "validation")],
        xgb_model=booster,  # continue training from previous model
        verbose_eval=10
    )

# === Predict on validation set ===
val_preds_proba = booster.predict(dval)
val_preds = np.argmax(val_preds_proba, axis=1)

print("\n=== Validation Performance ===")
print(classification_report(val_y, val_preds, target_names=label_encoder.classes_))
