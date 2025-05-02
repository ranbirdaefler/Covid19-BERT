# src/final_evaluate_and_compare.py

import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def load_data(data_dir, vector_type='bert'):
    if vector_type == 'bert':
        X = np.load(os.path.join(data_dir, 'diff_vectors.npy'))
    elif vector_type == 'static':
        X = np.load(os.path.join(data_dir, 'static_vectors.npy'))
    else:
        raise ValueError("vector_type must be 'bert' or 'static'")
    y_raw = np.load(os.path.join(data_dir, 'labels.npy'))
    return X, y_raw

def load_model(model_dir):
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, 'xgboost_model.json'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    return model, label_encoder

def evaluate(model, label_encoder, X, y_raw, dataset_name, model_name):
    y_true = label_encoder.transform(y_raw)
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {model_name} on {dataset_name} Set ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:\n")
    print(cm)

def main():
    # Paths
    bert_model_dir = "models/bert_xgboost"
    static_model_dir = "models/sparse_xgboost"

    # Load models
    bert_model, bert_label_encoder = load_model(bert_model_dir)
    static_model, static_label_encoder = load_model(static_model_dir)

    # --- Validation Set ---
    print("\nEvaluating on Validation Set...")

    X_val_bert, y_val_bert = load_data("embeddings/val", vector_type='bert')
    X_val_static, y_val_static = load_data("static_features/val", vector_type='static')

    evaluate(bert_model, bert_label_encoder, X_val_bert, y_val_bert, dataset_name="Validation", model_name="BERT XGBoost")
    evaluate(static_model, static_label_encoder, X_val_static, y_val_static, dataset_name="Validation", model_name="Static XGBoost")

    # --- Test Set ---
    print("\nEvaluating on Test Set...")

    X_test_bert, y_test_bert = load_data("embeddings/test", vector_type='bert')
    X_test_static, y_test_static = load_data("static_features/test", vector_type='static')

    evaluate(bert_model, bert_label_encoder, X_test_bert, y_test_bert, dataset_name="Test", model_name="BERT XGBoost")
    evaluate(static_model, static_label_encoder, X_test_static, y_test_static, dataset_name="Test", model_name="Static XGBoost")

if __name__ == "__main__":
    main()
