#!/usr/bin/env python
"""
variant_distance_matrix.py
--------------------------
Compute a lineage-to-lineage mean Hamming-distance matrix and plot it.
"""

import os, glob
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- CONFIG ------------------------------------------------
FASTA_DIR   = "/Users/floriandaefler/Desktop/deep and reinforcement/project/data"          # folder with alpha.fasta, beta.fasta, ...
MAX_PER_VAR = 1000            # down-sample cap per variant
SEED        = 0               # RNG for reproducibility
OUT_PNG     = "figs/variant_distance_matrix.png"
OUT_CSV     = "variant_distance_matrix.csv"
# --------------------------------------------------------------------------

os.makedirs("figs", exist_ok=True)
rng = np.random.default_rng(SEED)

def read_fasta_folder(folder):
    recs = {"seq": [], "variant": []}
    for fp in glob.glob(os.path.join(folder, "*.fasta")):
        variant = os.path.splitext(os.path.basename(fp))[0].capitalize()
        for rec in SeqIO.parse(fp, "fasta"):
            recs["seq"].append(str(rec.seq).upper())
            recs["variant"].append(variant)
    return pd.DataFrame(recs)

def downsample(df, k, rng):
    return (df.groupby("variant", group_keys=False)
              .apply(lambda x: x.sample(min(len(x), k), random_state=rng)))

def char_matrix(seq_series):
    """Return (N,L) uint8 array from pd.Series of equal-length strings."""
    return np.array([list(s.encode("ascii")) for s in seq_series], dtype="S1")

def mean_pairwise(mat_a, mat_b):
    """Average normalised Hamming distance between two groups."""
    # broadcasting trick: compare every row of A with every row of B
    diff = (mat_a[:, None, :] != mat_b[None, :, :]).mean(-1)
    return diff.mean()

def main():
    df = read_fasta_folder(FASTA_DIR)
    df = downsample(df, MAX_PER_VAR, rng)

    # enforce uniform length (modal spike length = 1 273)
    L_mode = df.seq.str.len().mode()[0]
    df = df[df.seq.str.len() == L_mode].reset_index(drop=True)
    seq_mat = char_matrix(df.seq)

    variants = sorted(df.variant.unique())
    idx_map  = {v:i for i,v in enumerate(df.index)}

    # build per-variant row indices
    rows = {v: np.where(df.variant == v)[0] for v in variants}

    # compute mean distances
    mat = np.zeros((len(variants), len(variants)))
    for i,v1 in enumerate(variants):
        for j,v2 in enumerate(variants):
            if i <= j:                    # upper triangle incl. diag
                d = mean_pairwise(seq_mat[rows[v1]], seq_mat[rows[v2]])
                mat[i,j] = mat[j,i] = d

    # save CSV
    pd.DataFrame(mat, index=variants, columns=variants).to_csv(OUT_CSV)

    # plot heat-map
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, fmt=".3f",
                xticklabels=variants, yticklabels=variants,
                cmap="mako_r", cbar_kws={'label':'Mean normalised\nHamming distance'})
    plt.title("Inter-variant spike-protein divergence")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()

    print(f"Matrix saved → {OUT_CSV}")
    print(f"Heat-map  → {OUT_PNG}")

if __name__ == "__main__":
    main()
