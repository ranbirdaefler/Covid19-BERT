# Evaluating Protein Language Models Against Traditional and Positional Features for SARS-CoV-2 Variant Classification

This repository contains code and resources for the project:

> **Evaluating Protein Language Models Against Traditional and Positional Features for SARS-CoV-2 Variant Classification**  
> *Florian Ranbir Vaid Daelfer, Emilija Milanovic, Changchen Yu*  
> Bocconi University Deep Learning and Reinforcement Learning

## Additional files that were too big to push 
https://limewire.com/d/swG4V#Nu8ABipUhL

This link leads to a download of the embeddings garnered via ProtBERT (as a .npy file) and the labels associated for classification. Additionally you can also find all the .fasta files (the ones from Kaggle) and the labels in CSV format.

## Overview

This project systematically compares three approaches to representing SARS-CoV-2 spike protein sequences for the task of variant classification:

1. **AAindex Aggregate Features**: Simple 6-dimensional vectors summarizing global hydrophobicity changes.
2. **AAindex Positional Features**: 1273-dimensional vectors capturing site-specific hydrophobicity mutations.
3. **ProtBERT CLS Embeddings**: High-dimensional learned representations from the [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert) protein language model, differenced against the Wuhan reference.

All approaches are evaluated on a large, diverse dataset of 261,042 spike protein sequences spanning six WHO lineages, using XGBoost classifiers with method-specific hyperparameter optimization and rigorous cross-validation.

## Key Findings

- **Positional encoding of mutations** outperforms both simple baselines and ProtBERT embeddings, achieving the highest macro-F1 (0.967) and accuracy (99.0%).
- **ProtBERT embeddings** significantly outperform the simple baseline, offering a strong "off-the-shelf" solution with minimal domain expertise required.
- **Carefully engineered features** still provide meaningful performance gains, especially for minority variant detection, despite advances in protein language models.

## Dataset

- **Source**: [Kaggle SARS-CoV-2 Spike Protein Sequences](https://www.kaggle.com/datasets/edumath/sars-cov-2-spike-sequences) (mirroring GISAID records).
- **Classes**: Alpha, Beta, Gamma, Delta, Omicron, Others (catch-all).
- **Sequence length**: All sequences are aligned to the Wuhan reference (NCBI accession YP_009724390.1).
- **Ambiguity Handling**: Non-standard residues (X, B, Z, J) are ignored during feature computation.

## Repository Structure

```
README.md
requirements.txt
data/
    alpha.fasta
    beta.fasta
    delta.fasta
    gamma.fasta
    omicron.fasta
    others.fasta
    reference.fasta
dataloader/
    compute_bert_embeddings.py
    extract_static_features.py
    loader.py
    static_feature_extractor.py
e.d.a/
    entropy_check.py
    hamming.py
    variant_distance_matrix.py
models/
    cv.py
```

### Directory/Script Descriptions

- **data/**: Contains FASTA files for each variant and the Wuhan reference.
- **dataloader/**: Scripts for loading data and extracting features(You can download the generated data via the first link in the readme):
  - `compute_bert_embeddings.py`: Generate BERT-based embeddings for spike sequences.
  - `extract_static_features.py`, `static_feature_extractor.py`: Extract AAindex-based features (aggregate and positional).
  - `loader.py`: Data loading utilities.
- **e.d.a/**: Exploratory Data Analysis scripts:
  - `entropy_check.py`: Sequence entropy analysis.
  - `hamming.py`: Hamming distance statistics.
  - `variant_distance_matrix.py`: Compute and visualize distance matrices between variants.
- **models/**: Model training and evaluation.
  - `cv.py`: Cross-validation, model selection, and evaluation routines.
- **requirements.txt**: Python dependencies for running the project.

> **Note:** The original structure referenced `feature_extraction/`, `classification/`, and `analysis/` directories, but the current repository layout consolidates these into `dataloader/`, `models/`, and `e.d.a/` respectively. Adjust your script paths accordingly.

## Feature Extraction Approaches

### 1. AAindex Aggregate Features (Simple Baseline)
- Encodes each sequence as a 6D vector of global hydrophobicity change statistics (mean, std, sum, max, min, mutation rate).
- Based on Kyte–Doolittle hydrophobicity scale.

### 2. AAindex Positional Features (Enhanced Baseline)
- Encodes each sequence as a 1273D vector, with each entry representing the hydrophobicity change at a specific position (mutation or 0).
- Explicitly preserves the spatial distribution of mutations.

### 3. ProtBERT Embeddings
- Uses [ProtBERT](https://huggingface.co/Rostlab/prot_bert), a transformer model pre-trained on >200M protein sequences.
- Each sequence is tokenized and passed through ProtBERT; the [CLS] token's 1024-dimensional embedding is extracted.
- Variant embeddings are differenced against the Wuhan reference for stronger signal.

## Classification Framework

- **Model**: XGBoost (multi-class, softprob).
- **Hyperparameter Optimization**: Per-approach tuning using macro-F1, with 3-fold cross-validation on stratified subsets.
- **Final Evaluation**: 5-fold stratified cross-validation on the full set (N = 261,042).
- **Metrics**: Accuracy and macro-F1 (primary). Per-class precision, recall, and F1 are also reported.

## Results

| Feature Set                | Accuracy           | Macro-F1           |
|----------------------------|-------------------|--------------------|
| **AAindex positional (1273D)** | **0.990 ± 0.0003** | **0.967 ± 0.002**  |
| ProtBERT Δ-embeddings (1024D)  | 0.987 ± 0.0003     | 0.959 ± 0.002      |
| AAindex aggregate (6D)         | 0.980 ± 0.0006     | 0.928 ± 0.003      |

- **AAindex positional features** excel, especially in minority variants (Beta, Omicron).
- **ProtBERT** is a strong, easy-to-deploy baseline, especially useful for "Others" or ambiguous classes.
- **Aggregate baseline** is outperformed by both, but still achieves high absolute performance.

## Usage

### Requirements

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- XGBoost
- NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

Install dependencies:
```bash
pip install -r requirements.txt
```

### Data

- Place the provided FASTA files in the `data/` directory.
- The Wuhan reference sequence is included as `reference.fasta`.

### Running Feature Extraction and Classification

1. **Feature Extraction**:  
   - ProtBERT embeddings: `dataloader/compute_bert_embeddings.py`
   - AAindex features: `dataloader/extract_static_features.py` or `dataloader/static_feature_extractor.py`

2. **Model Training & Cross-Validation**:  
   - Use `models/cv.py` for XGBoost training, hyperparameter tuning, and evaluation.

3. **Exploratory Data Analysis**:  
   - Scripts in `e.d.a/` allow for entropy analysis, hamming distance calculations, and variant distance matrix visualization.

#### Example
```bash
python dataloader/compute_bert_embeddings.py --input data/alpha.fasta --output features/protbert_alpha.npy
python dataloader/extract_static_features.py --input data/alpha.fasta --output features/aaindex_alpha.npy
python models/cv.py --features features/protbert_alpha.npy --labels data/labels.csv
```

## Figures

- **jsd_heatmap.png**: Visualizes average Jensen–Shannon divergence between variants.
- **hamming_distribution.png**: Distribution of normalized Hamming distances, within and between variants.
- **variant_distance_matrix.png**: Matrix of mean Hamming distances between variant pairs.

## Contact

For questions or issues, please open an [issue](https://github.com/ranbirdaefler/Covid19-BERT/issues) or contact Florian Ranbir Vaid Daelfer.

---

**This repository provides a reproducible benchmark for evaluating feature extraction approaches on large-scale SARS-CoV-2 variant classification. Contributions and suggestions are welcome!**
