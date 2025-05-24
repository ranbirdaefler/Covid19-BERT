#!/usr/bin/env python
"""
pairwise_hamming.py
-------------------
Compute and plot the distribution of pairwise Hamming distances
within and between SARS-CoV-2 lineages.
"""

import os
import glob
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------- CONFIG ---------------------------------
FASTA_DIR   = "/Users/floriandaefler/Desktop/deep and reinforcement/project/data"        # folder containing *.fasta
OUT_PNG     = "figs/hamming_distribution.png"
MAX_PER_VAR = 1000          # down-sample each lineage to this many sequences
SEED        = 0             # reproducibility
# --------------------------------------------------------------------

rng = np.random.default_rng(SEED)
os.makedirs("figs", exist_ok=True)

def read_fasta_folder(fasta_dir):
    """
    Returns a DataFrame with columns: seq (str), variant (str)
    Assumes filename encodes the variant, e.g. alpha.fasta -> Alpha
    """
    records = defaultdict(list)
    for fp in glob.glob(os.path.join(fasta_dir, "*.fasta")):
        variant = os.path.splitext(os.path.basename(fp))[0].capitalize()
        for rec in SeqIO.parse(fp, "fasta"):
            records["seq"].append(str(rec.seq).upper())
            records["variant"].append(variant)
    return pd.DataFrame(records)

def downsample(df, max_per_var, rng):
    return (df.groupby("variant", group_keys=False)
              .apply(lambda x: x.sample(min(len(x), max_per_var),
                                        random_state=rng)))
    
def pairwise_distances(seq_array, labels):
    """
    Returns two 1-D numpy arrays: within, between (normalised distances)
    """
    n, L = seq_array.shape
    within, between = [], []
    for (i, lab_i), (j, lab_j) in combinations(enumerate(labels), 2):
        d = (seq_array[i] != seq_array[j]).sum() / L
        (within if lab_i == lab_j else between).append(d)
    return np.array(within), np.array(between)

def main():
    # 1) Load and down-sample
    df = read_fasta_folder(FASTA_DIR)
    print(f"Loaded {len(df):,} sequences across {df.variant.nunique()} variants")

    df = downsample(df, MAX_PER_VAR, rng)
    print(f"Down-sampled to {len(df):,} sequences "
          f"({MAX_PER_VAR} per variant max)")

    # 2) Keep only the modal length (aligned sequences)
    seq_lengths = df.seq.str.len()
    L_mode = seq_lengths.mode()[0]
    df = df[seq_lengths == L_mode].reset_index(drop=True)
    print(f"Using {len(df):,} sequences of uniform length {L_mode}")

    # 3) Convert to a (N, L) char-matrix
    seq_mat = np.array([list(s.encode("ascii")) for s in df.seq],
                       dtype="S1")   # shape = (N, L_mode)

    # 4) Compute distances
    within, between = pairwise_distances(seq_mat, df.variant.values)
    print(f"Computed {len(within):,} within and {len(between):,} between distances")

    # 5) Plot
    plt.figure(figsize=(7,4))
    sns.histplot(within, bins=60, stat="density",
                 alpha=0.8, label="within lineage", color="steelblue")
    sns.histplot(between, bins=60, stat="density",
                 alpha=0.5, label="between lineages", color="darkorange")
    plt.xlabel("Normalised Hamming distance")
    plt.ylabel("Density")
    plt.title("Pairwise spike-protein distances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"Saved histogram → {OUT_PNG}")


if __name__ == "__main__":
    main()
