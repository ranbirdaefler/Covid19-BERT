import os
from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split

class SpikeMultiDataLoaderWithReference:
    def __init__(self, data_dir, reference_path, min_length=1000, max_length=1500):
        self.data_dir = data_dir
        self.reference_path = reference_path
        self.min_length = min_length
        self.max_length = max_length

    def load_reference(self):
        """
        Loads the reference spike protein sequence from a FASTA file.
        """
        record = next(SeqIO.parse(self.reference_path, "fasta"))
        reference_seq = str(record.seq)
        print(f"Loaded reference sequence of length {len(reference_seq)}.")
        return reference_seq

    def load_all_data(self):
        """
        Loads all variant sequences and attaches the reference sequence to each entry.
        """
        reference_seq = self.load_reference()

        all_data = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".fasta") and filename != os.path.basename(self.reference_path):
                variant = filename.replace(".fasta", "").strip()
                filepath = os.path.join(self.data_dir, filename)
                records = list(SeqIO.parse(filepath, "fasta"))

                for record in records:
                    accession = record.id
                    sequence = str(record.seq)
                    if self.min_length <= len(sequence) <= self.max_length:
                        all_data.append({
                            "Sequence": sequence,
                            "Label": variant,
                            "AccessionID": accession,
                            "ReferenceSequence": reference_seq
                        })

        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} variant sequences across {df['Label'].nunique()} variants.")
        self.df = df
        return df

    def split_data(self, test_size=0.1, val_size=0.1, random_state=42):
        df_train, df_temp = train_test_split(self.df, test_size=(test_size + val_size), stratify=self.df["Label"], random_state=random_state)
        df_val, df_test = train_test_split(df_temp, test_size=test_size / (test_size + val_size), stratify=df_temp["Label"], random_state=random_state)

        print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

#below you must update the data_dir and reference_path to point to the folder with the .fasta files for the former and the reference wuhan path for the latter
loader = SpikeMultiDataLoaderWithReference(
    r'C:\Users\avsd8\OneDrive\Desktop\covid19\project\data',
    r'C:\Users\avsd8\OneDrive\Desktop\covid19\project\data\reference.fasta'
)
df = loader.load_all_data()
train_df, val_df, test_df = loader.split_data()

# Show basic info
print(df.head())

# How many sequences per variant?
print(df["Label"].value_counts())

# Check average sequence length
df["SequenceLength"] = df["Sequence"].str.len()
print(df["SequenceLength"].describe())
