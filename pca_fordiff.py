import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# === Load data ===
X = np.load("embeddings/train/diff_vectors.npy")
y_raw = np.load("embeddings/train/labels.npy")

# === Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
label_names = label_encoder.classes_

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

X_sample = X[sample_indices]
y_sample = y[sample_indices]
labels_sample = [label_names[i] for i in y_sample]

# === PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_sample)

# === Plotly 3D Plot
fig = px.scatter_3d(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    z=X_pca[:, 2],
    color=labels_sample,
    title="3D PCA of COVID Variant Embeddings (Pre-Fine-Tuning)",
    labels={"color": "Variant"},
    opacity=0.7
)

fig.update_traces(marker=dict(size=4))
fig.show()
