import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

# === Config
MODEL_PATH = "models/protbert_covid_finetuned.pt"
MODEL_NAME = "Rostlab/prot_bert"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data"
SAMPLE_PER_VARIANT = 500

# === Load model without classifier head
class ProtBERTNoHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # CLS token

model = ProtBERTNoHead().to(DEVICE)
model.bert.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
model.eval()

# === Load data
from loader import SpikeMultiDataLoaderWithReference
loader = SpikeMultiDataLoaderWithReference(
    os.path.join(DATA_PATH),
    os.path.join(DATA_PATH, "reference.fasta")
)
df = loader.load_all_data()

# === Subsample 500 per variant (excluding "others")
# === Subsample once consistently (excluding "others") ===
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

# === Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

# === Embed sequences and reference
def tokenize(seqs):
    return tokenizer(
        [" ".join(list(seq)) for seq in seqs],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

diff_vectors = []
labels = []

reference_seq = sampled_df["ReferenceSequence"].iloc[0]
ref_tokens = tokenize([reference_seq])
ref_input_ids = ref_tokens["input_ids"].to(DEVICE)
ref_attention_mask = ref_tokens["attention_mask"].to(DEVICE)
ref_embedding = model(ref_input_ids, ref_attention_mask).squeeze()

for i in tqdm(range(len(sampled_df))):
    seq = sampled_df.iloc[i]["Sequence"]
    label = sampled_df.iloc[i]["Label"]

    tokens = tokenize([seq])
    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    variant_embedding = model(input_ids, attention_mask).squeeze()
    diff = (variant_embedding - ref_embedding).cpu().numpy()

    diff_vectors.append(diff)
    labels.append(label)

diff_vectors = np.array(diff_vectors)

# === PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(diff_vectors)

# === Plotly 3D
fig = px.scatter_3d(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    z=X_pca[:, 2],
    color=labels,
    title="3D PCA of COVID Variant Diff Embeddings (Fine-Tuned ProtBERT)",
    labels={"color": "Variant"},
    opacity=0.75
)
fig.update_traces(marker=dict(size=4))
fig.show()
