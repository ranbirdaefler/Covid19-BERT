# src/train_xgboost.py

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os

def train_xgboost(diff_vectors_path, labels_path, save_dir="models/sparse_xgboost"):
    # Load data
    X = np.load(diff_vectors_path)
    y_raw = np.load(labels_path)

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Initialize XGBoost classifier
    clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.3,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda',      # <-- GPU Training
        predictor='gpu_predictor',   # <-- GPU Inference
        objective='multi:softprob',  # Multiclass classification
        num_class=len(np.unique(y)), # Important!
        verbosity=2                  # Built-in XGBoost verbosity
    )

    # Train — simple version
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   # pass eval_metric directly
        verbose=True
    )

    # Final Validation Accuracy
    y_val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"\nFinal Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

    # Save model and label encoder
    os.makedirs(save_dir, exist_ok=True)
    clf.save_model(os.path.join(save_dir, "xgboost_model.json"))
    joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))
    print(f"Model and label encoder saved to {save_dir}")

if __name__ == "__main__":
    train_xgboost(
        diff_vectors_path="static_features/train/static_vectors.npy",
        labels_path="static_features/train/labels.npy",
        save_dir="models/sparse_xgboost"
    )
