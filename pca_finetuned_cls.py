import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import plotly.express as px

# === CONFIG ===
MODEL_NAME = "Rostlab/prot_bert"
MODEL_PATH = "models/protbert_covid_finetuned.pt"
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_VARIANT = 500
MAX_LEN = 512

# === Load Tokenizer and Model ===
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

class ProtBERTEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0]  # CLS token

embedder = ProtBERTEmbedder().to(DEVICE)
embedder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
embedder.eval()

# === Load Data ===
from loader import SpikeMultiDataLoaderWithReference
loader = SpikeMultiDataLoaderWithReference(
    os.path.join(DATA_DIR),
    os.path.join(DATA_DIR, "reference.fasta")
)
df = loader.load_all_data()

# === Subsample ===
SAMPLE_PER_VARIANT = 500
RANDOM_SEED = 42

variant_labels = df["Label"].unique()
sampled_df = []

for variant in variant_labels:
    if variant.lower() == "others":
        continue
    variant_df = df[df["Label"] == variant]
    sampled = variant_df.sample(n=min(SAMPLE_PER_VARIANT, len(variant_df)), random_state=RANDOM_SEED)
    sampled_df.append(sampled)

sampled_df = pd.concat(sampled_df).reset_index(drop=True)

# === Tokenize ===
def tokenize_sequence(seq):
    spaced_seq = " ".join(list(seq))
    tokens = tokenizer(spaced_seq, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
    return tokens["input_ids"], tokens["attention_mask"]

# === Embed Sequences ===
embeddings = []
labels = []

for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
    input_ids, attention_mask = tokenize_sequence(row["Sequence"])
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    cls_embed = embedder(input_ids, attention_mask).cpu().numpy().squeeze()
    embeddings.append(cls_embed)
    labels.append(row["Label"])

# === PCA ===
pca = PCA(n_components=3)
reduced = pca.fit_transform(np.vstack(embeddings))

# === Plot ===
pca_df = pd.DataFrame(reduced, columns=["PC1", "PC2", "PC3"])
pca_df["Label"] = labels

fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Label", opacity=0.7)
fig.update_layout(title="3D PCA of COVID Variant Embeddings", legend_title="Variant")
fig.show()
