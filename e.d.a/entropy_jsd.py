

import os, glob, itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS

# ------------------------------------------------------------
# CONFIGURATION
#below you must update the data_dir and reference_path to point to the folder with the .fasta files for the former and the reference wuhan path for the latter
# ------------------------------------------------------------

FASTA_DIR         = "/Users/floriandaefler/Desktop/deep and reinforcement/project/data"     # folder containing alpha.fasta, beta.fasta, ...
MAX_PER_VARIANT   = 1000       
ALPHABET          = list("ACDEFGHIKLMNPQRSTVWY") 
FIG_DIR           = "/Users/floriandaefler/Desktop/deep and reinforcement/project/figs" #OUTPUT FOLDER
os.makedirs(FIG_DIR, exist_ok=True)
rng = np.random.default_rng(0)
# ------------------------------------------------------------

# 1) ------------------- LOAD SEQUENCES -----------------------
variant_seqs = defaultdict(list)
for fp in glob.glob(os.path.join(FASTA_DIR, "*.fasta")):
    variant = os.path.splitext(os.path.basename(fp))[0].capitalize()
    if variant == "Reference":
        continue
    for rec in SeqIO.parse(fp, "fasta"):
        variant_seqs[variant].append(str(rec.seq).upper())

if not variant_seqs:
    raise RuntimeError(f"No *.fasta files found in {FASTA_DIR}")

# Determine the modal spike length across ALL sequences
modal_len = pd.Series([len(s) for v in variant_seqs for s in variant_seqs[v]]).mode()[0]
print(f"Modal spike length = {modal_len}")

# keep only sequences with that length, optionally down-sample
for var in list(variant_seqs):
    seqs = [s for s in variant_seqs[var] if len(s) == modal_len]
    if MAX_PER_VARIANT and len(seqs) > MAX_PER_VARIANT:
        seqs = rng.choice(seqs, size=MAX_PER_VARIANT, replace=False).tolist()
    variant_seqs[var] = seqs
    print(f"{var:<8s}: {len(seqs):,} sequences")

variants = sorted(variant_seqs)           # e.g. ['Alpha', 'Beta', ...]
L = modal_len

# 2) ------------------ BUILD MATRICES ------------------------
# matrices: dict  variant -> (N, L) numpy array of single-letter strings
matrices = {v: np.array([list(s) for s in seqs], dtype="U1")
            for v, seqs in variant_seqs.items()}

# 3) ------------ PER-POSITION PROBABILITY TABLES -------------
freq = {}      # variant -> (L, 20) matrix of probabilities
entropy = {}   # variant -> (L,)  per-position entropy  (optional)

for v, mat in matrices.items():
    P_rows, H_rows = [], []
    for j in range(L):
        col = mat[:, j]
        col = col[col != 'X']           # ignore unknown / gap
        total = len(col)
        if total == 0:
            p = np.zeros(len(ALPHABET))
        else:
            counts = Counter(col)
            p = np.array([counts.get(a, 0)/total for a in ALPHABET])
        P_rows.append(p)
        H_rows.append(-(p*np.log2(np.clip(p, 1e-12, 1))).sum())
    freq[v]    = np.vstack(P_rows)          # (L, 20)
    entropy[v] = np.array(H_rows)           # (L,)

# 4) ----------------- JSD FUNCTION ---------------------------
def jsd_matrix(PA, PB):
    """Return mean Jensen–Shannon divergence (bits) over all positions."""
    M = 0.5*(PA + PB)                                   # (L,20)
    kl_pm = (PA * (np.log2(np.clip(PA,1e-12,1)) -
                   np.log2(np.clip(M ,1e-12,1)))).sum(1)
    kl_qm = (PB * (np.log2(np.clip(PB,1e-12,1)) -
                   np.log2(np.clip(M ,1e-12,1)))).sum(1)
    return 0.5*(kl_pm + kl_qm).mean()                   # scalar

# 5) -------------- PAIRWISE JSD MATRIX -----------------------
jsd_mat = pd.DataFrame(index=variants, columns=variants, dtype=float)
for A, B in itertools.product(variants, variants):
    jsd_mat.loc[A, B] = jsd_matrix(freq[A], freq[B])

# 6) -------------- FIGURE 1: HEAT-MAP ------------------------
sns.set(style="white")
cg = sns.clustermap(jsd_mat.astype(float),
                    cmap="viridis", annot=True, fmt=".3f",
                    linewidths=.3, cbar_kws={'label':'Mean JSD (bits)'},
                    figsize=(6, 5))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
cg.fig.suptitle("Mean Jensen–Shannon divergence between variants",
                y=1.03, fontsize=12)
heatmap_fp = os.path.join(FIG_DIR, "jsd_heatmap.png")
cg.savefig(heatmap_fp, dpi=300)
plt.close()

# 7) -------------- FIGURE 2: 2-D MDS --------------------------
mds = MDS(dissimilarity="precomputed", random_state=0)
coords = mds.fit_transform(jsd_mat.values)   # (K,2)

plt.figure(figsize=(4,4))
for (x, y), lab in zip(coords, variants):
    plt.scatter(x, y, s=180)
    plt.text(x, y, lab, ha="center", va="center", weight="bold", color="white")
plt.title("Metric-MDS of Jensen–Shannon distances")
plt.axis("off")
plt.tight_layout()
mds_fp = os.path.join(FIG_DIR, "jsd_mds.png")
plt.savefig(mds_fp, dpi=300)
plt.close()

print(f"Figures written to {FIG_DIR}/")
