# src/static_feature_extractor.py

import pandas as pd
import numpy as np
import os

class StaticMutationExtractor:
    def __init__(self, aaindex_dict):
        self.aaindex = aaindex_dict
        self.amino_acids = list('ARNDCQEGHILKMFPSTWYV')  # Standard 20 AAs

    def get_aa_value(self, aa):
        # Return AAindex value, or 0.0 if unknown
        return self.aaindex.get(aa, 0.0)

    def compute_mutation_vector(self, ref_seq, var_seq):
        mutation_effects = []

        for r_aa, v_aa in zip(ref_seq, var_seq):
            if r_aa not in self.amino_acids or v_aa not in self.amino_acids:
                continue  # Ignore non-standard AAs
            if r_aa != v_aa:
                diff = self.get_aa_value(v_aa) - self.get_aa_value(r_aa)
                mutation_effects.append(diff)

        # Aggregate features
        if mutation_effects:
            feature_vector = [
                np.mean(mutation_effects),
                np.std(mutation_effects),
                np.sum(mutation_effects),
                np.max(mutation_effects),
                np.min(mutation_effects),
                len(mutation_effects) / len(ref_seq)  # mutation rate
            ]
        else:
            feature_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return np.array(feature_vector)

    def compute_features_for_dataframe(self, df):
        feature_vectors = []
        labels = []

        for idx, row in df.iterrows():
            ref_seq = row["ReferenceSequence"]
            var_seq = row["Sequence"]
            label = row["Label"]

            vector = self.compute_mutation_vector(ref_seq, var_seq)
            feature_vectors.append(vector)
            labels.append(label)

        return np.vstack(feature_vectors), np.array(labels)

def save_features(features, labels, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'static_vectors.npy'), features)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
