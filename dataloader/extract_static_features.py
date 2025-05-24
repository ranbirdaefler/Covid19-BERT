# src/extract_static_features.py

import pandas as pd
import numpy as np
import os
from static_feature_extractor import StaticMutationExtractor, save_features
from loader import SpikeMultiDataLoaderWithReference  # <-- import your loader

# Define AAindex (Kyte-Doolittle hydrophobicity scale)
aaindex_hydrophobicity = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def main():
    # Initialize the loader (below you must update the data_dir and reference_path to point to the folder with the .fasta files for the former and the reference wuhan path for the latter)
    loader = SpikeMultiDataLoaderWithReference(
        data_dir=r'C:\Users\avsd8\OneDrive\Desktop\covid19\project\data',
        reference_path=r'C:\Users\avsd8\OneDrive\Desktop\covid19\project\data\reference.fasta'
    )
    df = loader.load_all_data()
    train_df, val_df, test_df = loader.split_data()

    # Initialize static feature extractor
    extractor = StaticMutationExtractor(aaindex_hydrophobicity)

    # --- Process Train set ---
    features_train, labels_train = extractor.compute_features_for_dataframe(train_df)
    save_features(features_train, labels_train, "static_features/train")

    # --- Process Validation set ---
    features_val, labels_val = extractor.compute_features_for_dataframe(val_df)
    save_features(features_val, labels_val, "static_features/val")

    # --- Process Test set ---
    features_test, labels_test = extractor.compute_features_for_dataframe(test_df)
    save_features(features_test, labels_test, "static_features/test")

if __name__ == "__main__":
    main()
