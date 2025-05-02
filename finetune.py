import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm

# =======================
# Config
# =======================
DATA_DIR = "data"
BATCH_SIZE = 16
EPOCHS = 4
LR = 1e-5
MAX_LEN = 512
MODEL_NAME = "Rostlab/prot_bert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Pre-tokenize dataset
# =======================
class PreTokenizedCovidSpikeDataset(Dataset):
    def __init__(self, df, tokenizer, label_encoder):
        self.inputs = tokenizer(
            df["Sequence"].apply(lambda x: " ".join(list(x))).tolist(),
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(label_encoder.transform(df["Label"]), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'label': self.labels[idx]
        }

# =======================
# Model
# =======================
class ProtBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(cls_token)
        return logits

# =======================
# Training Function
# =======================
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    scaler = GradScaler('cuda')

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# =======================
# Main Execution Block
# =======================
if __name__ == "__main__":
    from loader import SpikeMultiDataLoaderWithReference
    loader = SpikeMultiDataLoaderWithReference(
        os.path.join(DATA_DIR),
        os.path.join(DATA_DIR, "reference.fasta")
    )
    df = loader.load_all_data()
    train_df, val_df, test_df = loader.split_data()

# Subsample 25% of training data
    train_df = train_df.sample(frac=0.10, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=0.10, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=0.10, random_state=42).reset_index(drop=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(df["Label"])

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    train_ds = PreTokenizedCovidSpikeDataset(train_df, tokenizer, label_encoder)
    val_ds = PreTokenizedCovidSpikeDataset(val_df, tokenizer, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = ProtBERTClassifier(num_classes=len(label_encoder.classes_))
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {DEVICE}...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/protbert_covid_finetuned.pt")
    print("Model saved.")
