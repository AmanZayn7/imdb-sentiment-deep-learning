# IMDB Sentiment Classification using BiLSTM and TextCNN (PyTorch)

## Project Overview

This project implements and rigorously compares two deep learning architectures — **BiLSTM** and **TextCNN** — for binary sentiment classification on the IMDB Large Movie Review dataset (50,000 reviews).

All models were trained under identical experimental conditions using disciplined preprocessing, stratified splitting, F1-driven early stopping, validation-based threshold tuning, and reproducible artifact handling.

The tuned **BiLSTM (Top-2 ensemble)** was selected as the final model.

# Dataset

- **Source:** IMDB Large Movie Review Dataset  
- **Total reviews:** 50,000  
- **Balanced:** 25,000 positive / 25,000 negative  

## Deduplication
- Exact duplicates removed: **418**
- Rough normalized duplicates removed: **422**

## Label Mapping
- `positive → 1`
- `negative → 0`


# Exploratory Data Analysis (EDA Highlights)

- **Median token length:** ≈ 173  
- **p90:** ≈ 451  
- **p95:** ≈ 590  

Lexical polarity indicators observed:

- Negative: `bad`, `worst`, `awful`
- Positive: `great`, `love`, `excellent`

These statistics informed vocabulary size and sequence length decisions.


# Data Pre-Processing Pipeline

## Text Normalization
- HTML entity trimming and tag removal
- Unicode normalization (NFKC)
- Lower-casing
- Repeated whitespace collapse

## Tokenization
- Punctuation-aware tokenizer
- Contractions preserved (e.g., *don't*, *isn't*)

## Stratified Split (Fixed Seed)

After deduplication:

- **Train:** 39,662  
- **Validation:** 4,958  
- **Test:** 4,958  

Split ratio: **80 / 10 / 10**

## Vocabulary Construction
- Training-only vocabulary (no leakage)
- Top **30,000** tokens retained
- Reserved indices:
  - `0 → [PAD]`
  - `1 → [UNK]`

## Sequence Standardization
- `MAX_LEN = 400`
- Optional truncation to 320 for MPS efficiency
- Integer-indexed sequences
- Strict no-leakage policy


# Model Architectures

## 1️⃣ TextCNN (Tuned)

### Architecture
- Embedding: (30,000 × 192)
- Parallel Conv1D layers (kernel sizes 3, 4, 5)
- Global max pooling
- Fully connected layer
- Dropout
- Sigmoid output

### Validation
- Best threshold: **0.54**
- Validation F1: **0.9010**

### Test Performance
- **Test Accuracy:** 0.8983
- **Test Macro F1:** 0.8983

**Class 0 (Negative)**  
- Precision: 0.9086  
- Recall: 0.8850  
- F1: 0.8966  

**Class 1 (Positive)**  
- Precision: 0.8887  
- Recall: 0.9116  
- F1: 0.9000  

**Confusion Matrix**
[[2186, 284],
[220, 2268]]


## 2️⃣ BiLSTM (Tuned — Best Single Snapshot)

### Architecture
- Embedding (30k, 192)
- BiLSTM (hidden=128, bidirectional)
- AdaptiveMaxPool1d
- Linear(256→128)
- Dropout(0.5)
- Linear(128→1)
- Sigmoid

### Optimization
- AdamW (`lr = 1e-3`, `weight_decay = 1e-5`)
- BCELoss
- Cosine learning rate schedule with warm-up
- Gradient accumulation
- Gradient clipping (`max_norm=1.0`)
- Early stopping based on validation F1 (`PATIENCE=1`)

### Validation
- Best threshold: **0.49**
- Validation F1: **0.9071**
- Validation Accuracy: **0.9054**

### Test Performance (Best Single)
- **Test Accuracy:** 0.9062
- **Test Macro F1:** 0.9062

**Class 0 (Negative)**  
- Precision: 0.9203  
- Recall: 0.8887  
- F1: 0.9042  

**Class 1 (Positive)**  
- Precision: 0.8931  
- Recall: 0.9236  
- F1: 0.9081  

**Confusion Matrix**
[[2195, 275],
[190, 2298]]



## 3️⃣ BiLSTM (Top-2 Ensemble — Final Model)

Two best validation-F1 checkpoints retained and ensembled (mean probability).

### Test Performance (Final Model)
- **Test Accuracy:** 0.9072
- **Test Macro F1:** 0.9072

**Class 0**
- Precision: 0.9187  
- Recall: 0.8927  
- F1: 0.9055  

**Class 1**
- Precision: 0.8964  
- Recall: 0.9216  
- F1: 0.9088  

**Confusion Matrix**
[[2205, 265],
[195, 2293]]

---

# Comparative Summary

- BiLSTM outperforms TextCNN on all headline metrics.
- Absolute improvement: ~0.8–0.9 percentage points.
- BiLSTM reduces false negatives for positives (190 vs 220 in CNN).
- Validation F1:
  - TextCNN: 0.9010
  - BiLSTM: 0.9071
- Threshold tuning:
  - TextCNN: 0.54
  - BiLSTM: 0.49

The Top-2 ensemble produces a small but consistent lift over the best single checkpoint.


# Final Model Selection

The tuned **BiLSTM (Top-2 ensemble)** was selected as the final model due to:

- Highest test accuracy (**0.9072**)
- Highest macro F1 (**0.9072**)
- Stronger recall for positive class
- Improved error symmetry
- Stable early-stopped learning curves

TextCNN remains a computationally efficient baseline suitable for lower-latency deployment.


# Technical Stack

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
