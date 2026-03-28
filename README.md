# 💳 Credit Card Fraud Detection

> A complete machine learning pipeline for detecting fraudulent credit card transactions — covering EDA, data preparation, unsupervised clustering, supervised classification, and neural networks on a highly imbalanced real-world dataset.

---

## 📌 Overview

This project was developed as a case study for the course **Ingegneria della Conoscenza** (University of Bari, A.Y. 2024–25). It addresses the problem of **binary classification on a severely imbalanced dataset** of European credit card transactions, applying and comparing multiple learning strategies to maximize fraud detection performance.

**Author:** Raffaele Gatta — Matricola 746397
**Repository:** [GitHub](https://github.com/raffytorres96/Caso-di-studio-Icon3/tree/main)
**Course:** Ingegneria della Conoscenza

---

## 🗂️ Table of Contents

1. [Dataset](#-dataset)
2. [Project Pipeline](#-project-pipeline)
3. [Exploratory Data Analysis](#-exploratory-data-analysis)
4. [Data Preparation](#-data-preparation)
5. [Handling Class Imbalance & Overfitting](#-handling-class-imbalance--overfitting)
6. [Unsupervised Learning — K-Means Clustering](#-unsupervised-learning--k-means-clustering)
7. [Supervised Classification](#-supervised-classification)
8. [Neural Network](#-neural-network)
9. [Results Summary](#-results-summary)
10. [Tech Stack](#-tech-stack)
11. [References](#-references)

---

## 📊 Dataset

The dataset contains credit card transactions made by **European cardholders in September 2013**.

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Fraud rate | **0.172%** |
| Time window | 2 consecutive days (172,792 seconds) |
| Feature type | All numerical |

### Features

| Feature | Type | Description |
|---|---|---|
| `V1`–`V28` | Float | PCA-transformed components (anonymized for privacy) |
| `Time` | Integer | Seconds elapsed from the first transaction |
| `Amount` | Integer | Transaction amount |
| `Class` | Binary | Target variable — `1` = Fraud, `0` = Legitimate |

> The PCA transformation assumes prior scaling of all `V1`–`V28` features. `Time` and `Amount` are the only non-transformed features and are scaled manually during preparation.

---

## 🔄 Project Pipeline

```
Raw Dataset (284,807 transactions)
        │
        ▼
Exploratory Data Analysis
  - Distribution analysis (Time, Amount, Class)
  - Temporal patterns of fraud vs legitimate
  - KDE plots across all V1–V28 features
        │
        ▼
Data Preparation
  - Missing value check (none found)
  - Zero-amount transaction handling (kept as valid)
  - StandardScaler on Time and Amount
        │
        ▼
Overfitting Mitigation Strategy
  - Full Dataset training
  - Sub-sampling (50/50 balanced subset)
  - 5-Fold Stratified Cross Validation
        │
        ▼
Unsupervised Analysis (K-Means, k=4 and k=7)
        │
        ▼
Supervised Classification (7 models × 3 training strategies)
        │
        ▼
Neural Network (SGD & RMSprop, full dataset & subsample)
        │
        ▼
Final Evaluation (AUC-ROC, Precision, Recall, F1)
```

---

## 🔍 Exploratory Data Analysis

### Class distribution

The dataset is **severely imbalanced**: fraudulent transactions represent only **0.172%** of all records.

```
Legitimate:  284,315  (99.827%)
Fraudulent:      492   (0.173%)
```

### Temporal patterns

Legitimate transactions follow a clear **circadian rhythm**, dropping significantly during European nighttime hours. Fraudulent transactions are distributed **uniformly across time**, with no clear time-of-day dependency — a key distinguishing signal.

### KDE feature analysis

KDE density plots across all 30 features reveal the following separability:

| Separability | Features |
|---|---|
| **Clearly separated** | V4, V11 |
| **Partially separated** | V12, V14 |
| **Distinctive profiles** | V1, V2, V3, V10 |
| **Very similar profiles** | V25, V26, V28 |

Legitimate transactions (Class=0) are generally centered around `0` for most PCA features, while fraudulent ones (Class=1) show **asymmetric, skewed distributions**.

---

## 🛠️ Data Preparation

**Missing values:** None detected across all columns.

**Zero-amount transactions:** 1,798 legitimate and 27 fraudulent transactions have `Amount = 0`. These were retained as valid after research confirmed that zero-amount transactions exist in real-world contexts (merchant card verification, lottery websites, etc.).

**Feature scaling:** `Time` and `Amount` are scaled using `StandardScaler` to align their range with the PCA-transformed `V1`–`V28` features, preventing distance-based models from being dominated by these two columns.

---

## ⚖️ Handling Class Imbalance & Overfitting

Training directly on the full imbalanced dataset causes models to predict the majority class (legitimate) with near-perfect accuracy while failing to detect fraud. Three strategies were applied and compared:

| Strategy | Description |
|---|---|
| **Full Dataset** | All 284,807 records, original class distribution |
| **Sub-Sampling (50/50)** | Balanced subset with equal fraud/legitimate ratio |
| **5-Fold Stratified Cross Validation** | Preserves original class proportions in each fold |

**Hyperparameter tuning** was performed using **Grid Search** for all models trained on the full dataset and the subsample. Cross-validation models were run with default or simplified hyperparameters due to high computational cost.

---

## 🔵 Unsupervised Learning — K-Means Clustering

K-Means was applied to identify potential sub-groups within the two main classes (not to replace the class labels, but to explore hidden structure).

**K selection via Elbow Method** (k range: 3–10): both `k=4` and `k=7` appeared as candidate elbow points.

### Results

| k | Finding |
|---|---|
| `k=4` | Labels distributed nearly uniformly across both classes — inconclusive |
| `k=7` | **Label 1** showed ~40% fraud ratio vs ~2% in legitimate class — a significant signal |

Despite this finding, cluster labels from k=7 were **not included** in the supervised phase, as the pattern was localized to a single label and did not generalize.

---

## 🤖 Supervised Classification

Seven classifiers were trained and evaluated across the three data strategies.

### Models

| Model | Key Hyperparameters (Full Dataset) |
|---|---|
| **KNN** | `metric=manhattan`, `n_neighbors=5`, `weights=distance` |
| **Decision Tree** | `max_depth=5`, `min_samples_split=5` |
| **Random Forest** | `max_depth=10`, `n_estimators=300`, `max_features=sqrt` |
| **AdaBoost** | `algorithm=SAMME.R`, `learning_rate=0.5`, `n_estimators=200` |
| **Gradient Boosting** | `learning_rate=0.1`, `max_depth=3`, `n_estimators=200` |
| **Logistic Regression** | `C=10`, `penalty=l2`, `solver=lbfgs` |
| **Gaussian Naive Bayes** | Default (insensitive to hyperparameters) |

### AUC-ROC Comparison

#### Full Dataset

| Model | AUC |
|---|---|
| Random Forest | **0.98** |
| AdaBoost | 0.98 |
| Logistic Regression | 0.97 |
| KNN | 0.94 |
| Decision Tree | 0.92 |
| Gradient Boosting | 0.79 |

> ⚠️ Full dataset training shows overfitting: Class 0 metrics consistently at 1.00 while Class 1 Recall is ~0.74. The imbalance dominates.

#### Sub-Sample (50/50)

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | AUC |
|---|---|---|---|---|
| **AdaBoost** | 0.962 | 0.874 | 0.916 | 0.94 |
| **GradientBoosting** | 0.974 | 0.862 | 0.915 | 0.97 |
| **Random Forest** | 1.000 | 0.839 | 0.913 | 0.98 |
| **Decision Tree** | 1.000 | 0.851 | 0.919 | 0.95 |
| KNN | 0.949 | 0.851 | 0.897 | 0.97 |
| Logistic Regression | 0.974 | 0.862 | 0.915 | 0.97 |

> ✅ Sub-sampling significantly improves Recall and F1 for fraud detection. Decision Tree and Random Forest achieve **Precision = 1.00** on fraudulent transactions.

#### 5-Fold Stratified Cross Validation

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **Random Forest** | 0.951 | 0.781 | 0.857 | **0.95** |
| **AdaBoost** | 0.809 | 0.687 | 0.742 | 0.96 |
| KNN | 0.935 | 0.774 | 0.847 | 0.93 |
| Decision Tree | 0.743 | 0.778 | 0.760 | 0.89 |
| Gradient Boosting | 0.769 | 0.603 | 0.661 | 0.76 |
| Naive Bayes | 0.062 | 0.829 | 0.115 | — |

> ⚠️ Cross validation confirms moderate overfitting in Gradient Boosting (train vs test accuracy gap). Naive Bayes shows high recall but very low precision — unreliable for production use.

---

## 🧠 Neural Network

A **feedforward neural network** was implemented with TensorFlow/Keras for binary classification.

### Architecture

```
Input Layer     →  30 units (one per feature), activation: ReLU
Hidden Layer 1  →  10 units, activation: ReLU
Hidden Layer 2  →   8 units, activation: ReLU
Output Layer    →   1 unit,  activation: Sigmoid (fraud probability)
```

**Loss function:** `binary_crossentropy`
**Training:** 100 epochs, batch size 64

### Optimizer Comparison

| Optimizer | Dataset | Learning Rate | Notes |
|---|---|---|---|
| SGD | Full dataset | 0.01 | Higher rates caused divergence |
| SGD | Subsample | 0.1 | More aggressive learning stable on balanced data |
| RMSprop | Full dataset | adaptive | More stable than SGD; dynamic rate adaptation |

### Architecture depth choice

Testing both 2-layer (10, 8 units) and 3-layer (10, 8, 6 units) configurations revealed comparable metrics but **higher loss** in the deeper network. The **2-layer architecture** was selected for lower complexity at equivalent performance.

### Best Neural Network Results

| Configuration | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|
| SGD · Subsample · lr=0.1 | 0.98 | 0.88 | 0.93 |
| SGD · Full dataset · lr=0.01 | 0.87 | 0.77 | 0.82 |
| RMSprop · Full dataset | 0.77 | 0.77 | 0.77 |

> ✅ Neural networks produced the **most homogeneous results** across all training configurations compared to probabilistic models.

---

## 📈 Results Summary

| Model | Strategy | Recall (fraud) | F1 (fraud) | AUC | Notes |
|---|---|---|---|---|---|
| **Neural Network (SGD)** | Subsample | **0.88** | **0.93** | — | Most balanced overall |
| Random Forest | Subsample | 0.84 | 0.91 | 0.98 | Precision = 1.00 |
| AdaBoost | Subsample | 0.87 | 0.92 | 0.94 | Strong ensemble |
| Random Forest | Cross-Val | 0.78 | 0.86 | 0.95 | Best CV model |
| Logistic Regression | Subsample | 0.86 | 0.91 | 0.97 | Efficient baseline |
| Naive Bayes | Any | 0.82 | 0.11 | — | High FP rate — unreliable |

**Winner:** The **Neural Network trained on the subsample with SGD (lr=0.1)** delivered the most consistent and reliable results across all metrics. Random Forest on subsample is the best-performing classical model.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Classical ML models, GridSearch, Cross-Validation |
| TensorFlow / Keras | Neural network implementation |
| Matplotlib / Seaborn | EDA and result visualization |

---

## 📚 References

1. Dal Pozzolo, A. et al. *"Calibrating Probability with Undersampling for Unbalanced Classification"*. IEEE SSCI, 2015.
2. Murphy, K. P. *"Machine Learning: A Probabilistic Perspective"*. MIT Press, 2012.
3. Chollet, F. *"Deep Learning with Python"*. Manning Publications, 2017.

---

## 👤 Author

**Raffaele Gatta**
Computer Science — University of Bari
Course: Ingegneria della Conoscenza

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/raffytorres96/Caso-di-studio-Icon3)

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
</p>
