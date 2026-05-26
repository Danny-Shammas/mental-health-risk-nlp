# 🧠 Mental Health Risk Detection using NLP

A deep learning project that classifies Reddit posts into mental health categories using fine-tuned DistilBERT.

**Course:** Neural Networks and Deep Learning  
**Dataset:** [Mental Disorders Identification — Reddit NLP (Kaggle)](https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp)

---

## 📋 Project Overview

This project builds a text classification system that reads Reddit posts and identifies which mental health condition the post relates to. It compares a simple baseline (TF-IDF + Logistic Regression) against a fine-tuned transformer model (DistilBERT).

### Classes Detected
- Depression
- Anxiety
- BPD (Borderline Personality Disorder)
- Bipolar
- Schizophrenia
- Mental Illness (general)

---

## 🗂️ Project Structure

```
Step 1  → Environment setup & GPU check
Step 2  → Data loading from Kaggle (701,787 posts)
Step 3  → Text cleaning & balanced sampling (60k/class)
Step 4  → Train / Validation / Test split (80/10/10, stratified)
Step 5  → Tokenization & PyTorch DataLoaders
Step 6  → DistilBERT model setup (66M parameters)
Step 7  → Model training with early stopping
Step 8  → Learning curves visualization
Step 9  → Baseline comparison (TF-IDF + Logistic Regression)
Step 10 → Experiment A: Title vs Body vs Combined (DistilBERT)
Step 11 → Experiment B: Concept clustering (K-Means + UMAP)
Step 12 → Final evaluation: Confusion matrix & classification report
Step 13 → Prediction system with confidence scores
Step 14 → Attention visualization (Explainable AI)
Step 15 → Ethical analysis & bias detection
Step 16 → Error analysis
Step 17 → Save all results
Step 18 → Final project summary
```

---

## 🤖 Model

| Component | Detail |
|---|---|
| Base model | `distilbert-base-uncased` |
| Parameters | 66 million |
| Max sequence length | 128 tokens |
| Batch size | 32 |
| Epochs | 4 |
| Learning rate | 1e-5 (with warmup) |
| Optimizer | AdamW (weight decay 0.01) |
| Scheduler | Linear warmup + decay |
| Gradient clipping | max_norm = 1.0 |

---

## 📊 Results

| Model | Val Accuracy |
|---|---|
| TF-IDF + Logistic Regression (baseline) | ~70% |
| DistilBERT (fine-tuned) | ~75% |

### Experiment A — Input Comparison (using trained DistilBERT)
The same trained model was tested on different input types without retraining:

| Input | Accuracy |
|---|---|
| Title only | lower |
| Body (selftext) only | medium |
| Combined (title + body) | **best** |

### Experiment B — Concept Clustering
DistilBERT embeddings (CLS token) were extracted and clustered using K-Means, then visualized in 2D using UMAP. This shows what internal representations the model learned for each mental health condition.

---

## ⚖️ Ethical Analysis

This project includes a dedicated ethical analysis section that measures:
- **False Negative Rate (FNR)** per class — how many real cases the model misses
- **False Positive Rate (FPR)** per class — how many false alarms the model raises

In mental health AI, a missed case (false negative) is more dangerous than a false alarm — the analysis highlights which classes carry the highest risk.

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run All: `Runtime → Run All`
4. Results are saved automatically and downloaded as a zip file

> ⏱️ Estimated training time: ~2.5–3 hours on Colab T4 GPU

---

## 📦 Requirements

```
transformers
torch
scikit-learn
pandas
numpy
matplotlib
seaborn
umap-learn
kagglehub
```

All installed automatically in the first cell.

---

## 📁 Output Files

After running, the following files are saved:

| File | Description |
|---|---|
| `best_model/` | Saved DistilBERT checkpoint |
| `classification_report.txt` | Per-class precision, recall, F1 |
| `confusion_matrix.png` | Visual confusion matrix |
| `learning_curves.png` | Train loss and accuracy per epoch |
| `clustering_umap.png` | UMAP visualization of embeddings |
| `ethical_analysis.png` | FNR/FPR analysis per class |
| `error_analysis.png` | Most common misclassifications |
| `training_log.csv` | Epoch-by-epoch metrics |
| `test_predictions.csv` | All test predictions vs true labels |
| `experiment_a_results.csv` | Title vs body vs combined results |

---

## ⚠️ Disclaimer

This project is for **educational purposes only.** The model is not a clinical tool and should not be used for diagnosing or screening mental health conditions in real-world settings.
