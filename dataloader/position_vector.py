
"""

Create a 1 273-D per-site Kyte–Doolittle Δ-vector for every spike sequence.

Outputs
  ├─ positional_vectors.npy   # shape (N, 1273) float32
  └─ labels.npy               # shape (N,)       str
"""

# ------------------------------------------------------------------
# 1  Imports & constants
# ------------------------------------------------------------------
import os
import numpy as np
from tqdm import tqdm

from loader import SpikeMultiDataLoaderWithReference  # your existing loader

AAINDEX_HD = {            # Kyte–Doolittle hydrophobicity
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5,  'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
    'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}
STD_AA = set(AAINDEX_HD)

L = 1273    # canonical spike length

# ------------------------------------------------------------------
# 2  Per-sequence Δ vector (trunc+pad)
# ------------------------------------------------------------------
from itertools import zip_longest

def delta_vector_1273(ref_seq: str, var_seq: str) -> np.ndarray:
    """
    Return a float32 vector of length 1 273.
    • Truncates var_seq if longer, pads with zeros if shorter.
    """
    delta = np.zeros(L, dtype=np.float32)

    for j, (r, v) in enumerate(zip_longest(ref_seq[:L], var_seq[:L], fillvalue=None)):
        if v is None or r is None or r == v:
            continue
        if r in STD_AA and v in STD_AA:
            delta[j] = AAINDEX_HD[v] - AAINDEX_HD[r]
    return delta

# ------------------------------------------------------------------
# 3  Load data, change the paths as yoy have them in your directory. DATA_DIR should point to the folders with the .fastas and ref_fp must point to the reference fasta
# ------------------------------------------------------------------
DATA_DIR = r""   
REF_FP   = os.path.join(DATA_DIR, "")

loader = SpikeMultiDataLoaderWithReference(DATA_DIR, REF_FP)
df = loader.load_all_data()          # already adds ReferenceSequence col
ref_seq = df.loc[0, "ReferenceSequence"]

print(f"Total sequences loaded: {len(df):,}")

# ------------------------------------------------------------------
# 4  Compute feature matrix
# ------------------------------------------------------------------
vectors = np.empty((len(df), L), dtype=np.float32)

for i, row in tqdm(df.iterrows(), total=len(df), desc="Computing Δ vectors"):
    vectors[i] = delta_vector_1273(ref_seq, row["Sequence"])

labels = df["Label"].to_numpy()

# ------------------------------------------------------------------
# 5  Save
# ------------------------------------------------------------------
np.save("positional_vectors.npy", vectors)
np.save("labels.npy", labels)
print("Saved positional_vectors.npy and labels.npy")
