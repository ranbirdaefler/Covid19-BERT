# src/train_mlp_bert.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

def load_data(embedding_dir):
    X = np.load(os.path.join(embedding_dir, 'diff_vectors.npy'))
    y_raw = np.load(os.path.join(embedding_dir, 'labels.npy'))
    return X, y_raw

def main():
    # Hyperparameters
    batch_size = 128
    epochs = 3000   # <-- now 500 epochs fixed
    lr = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    X_train, y_train_raw = load_data('embeddings/train')
    X_val, y_val_raw = load_data('embeddings/val')
    X_test, y_test_raw = load_data('embeddings/test')

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    num_classes = len(label_encoder.classes_)

    # Create datasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = MLP(input_dim=X_train.shape[1], num_classes=num_classes)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Validation (just to monitor, not for stopping)
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        val_acc = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {total_loss/len(train_loader):.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f}")

    # Save final model manually
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mlp_bert_final.pth")
    print("\nModel saved to models/mlp_bert_final.pth")

    # Test evaluation
    print("\nEvaluating on Test set...")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    main()
