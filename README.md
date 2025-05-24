# Evaluating Protein Language Models Against Traditional and Positional Features for SARS-CoV-2 Variant Classification

This repository contains code and resources for the project:

> **Evaluating Protein Language Models Against Traditional and Positional Features for SARS-CoV-2 Variant Classification**  
> *Florian Ranbir Vaid Daelfer, Emilija Milanovic, Changchen Yu*  
> Bocconi University Deep Learning and Reinforcement Learning

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

Install dependencies (example):
```bash
pip install torch transformers xgboost numpy pandas scikit-learn matplotlib seaborn
```

### Data

- Download spike sequences from [Kaggle](https://www.kaggle.com/datasets/edumath/sars-cov-2-spike-sequences) and place in a `data/` directory.
- The Wuhan reference sequence is available from [NCBI](https://www.ncbi.nlm.nih.gov/protein/YP_009724390.1?report=fasta).

### Running Feature Extraction and Classification

1. **Feature Extraction**:  
   - `feature_extraction/aggregate_features.py`  
   - `feature_extraction/positional_features.py`  
   - `feature_extraction/protbert_embeddings.py`

2. **Classification**:  
   - `classification/train_xgboost.py` (handles hyperparameter tuning and cross-validation for each feature set)

3. **Analysis and Visualization**:  
   - `analysis/` contains scripts for exploratory analysis, performance metrics, and generating plots.

Example:
```bash
python feature_extraction/protbert_embeddings.py --input data/Alpha.fasta --output features/protbert_alpha.npy
python classification/train_xgboost.py --features features/protbert_alpha.npy --labels data/labels.csv
```

## Repository Structure

```
data/                  # FASTA files and reference sequences
feature_extraction/    # Scripts for each feature extraction approach
classification/        # XGBoost training, hyperparameter tuning, evaluation
analysis/              # Exploratory analysis, plotting scripts
results/               # Output metrics, tables, and figures
jsd_heatmap.png        # Jensen–Shannon divergence heatmap (see report)
hamming_distribution.png # Hamming distance histogram
variant_distance_matrix.png # Per-variant Hamming distance matrix
README.md              # This file
report.pdf             # Full paper (compiled from LaTeX)
```

## Figures

- **jsd_heatmap.png**: Visualizes average Jensen–Shannon divergence between variants.
- **hamming_distribution.png**: Distribution of normalized Hamming distances, within and between variants.
- **variant_distance_matrix.png**: Matrix of mean Hamming distances between variant pairs.

## Contact

For questions or issues, please open an [issue](https://github.com/ranbirdaefler/Covid19-BERT/issues) or contact Florian Ranbir Vaid Daelfer.

---

**This repository provides a reproducible benchmark for evaluating feature extraction approaches on large-scale SARS-CoV-2 variant classification. Contributions and suggestions are welcome!**
