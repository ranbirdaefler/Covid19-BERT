# src/bert_embedder.py

import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

class ProtBERTEmbedder:
    def __init__(self, model_name='Rostlab/prot_bert', device=None, batch_size=16):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        print(f"Using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_sequence(self, seq):
        # ProtBERT expects amino acids separated by spaces
        return ' '.join(list(seq))

    def embed_batch(self, sequences):
        # Preprocess
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embeddings.cpu().numpy()

    def compute_diff_vectors(self, df):
        variant_embeddings = []
        reference_embeddings = []
        labels = []

        all_variant_seqs = df["Sequence"].tolist()
        all_reference_seqs = df["ReferenceSequence"].tolist()
        all_labels = df["Label"].tolist()

        for i in tqdm(range(0, len(df), self.batch_size)):
            variant_batch = all_variant_seqs[i:i+self.batch_size]
            reference_batch = all_reference_seqs[i:i+self.batch_size]

            variant_batch_preprocessed = [self.preprocess_sequence(seq) for seq in variant_batch]
            reference_batch_preprocessed = [self.preprocess_sequence(seq) for seq in reference_batch]

            variant_embeds = self.embed_batch(variant_batch_preprocessed)
            reference_embeds = self.embed_batch(reference_batch_preprocessed)

            variant_embeddings.append(variant_embeds)
            reference_embeddings.append(reference_embeds)
            labels.extend(all_labels[i:i+self.batch_size])

        # Stack all batches
        variant_embeddings = np.vstack(variant_embeddings)
        reference_embeddings = np.vstack(reference_embeddings)

        diff_vectors = variant_embeddings - reference_embeddings

        print(f"Computed {len(diff_vectors)} diff vectors.")
        return diff_vectors, labels

    def save_embeddings(self, diff_vectors, labels, output_dir='embeddings/'):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "diff_vectors.npy"), diff_vectors)
        np.save(os.path.join(output_dir, "labels.npy"), np.array(labels))
        print(f"Saved diff vectors and labels to {output_dir}")

#Example usage:
from loader import SpikeMultiDataLoaderWithReference
loader = SpikeMultiDataLoaderWithReference(
    r'C:\Users\avsd8\OneDrive\Desktop\covid19\project\data',
    r'C:\Users\avsd8\OneDrive\Desktop\covid19\project\data\reference.fasta'
)

df = loader.load_all_data()
train_df, val_df, test_df = loader.split_data()

embedder = ProtBERTEmbedder(batch_size=64)
diff_vectors_val, labels_val = embedder.compute_diff_vectors(val_df)
embedder.save_embeddings(diff_vectors_val, labels_val, output_dir='embeddings/val')

diff_vectors_test, labels_test = embedder.compute_diff_vectors(test_df)
embedder.save_embeddings(diff_vectors_test, labels_test, output_dir='embeddings/test')

