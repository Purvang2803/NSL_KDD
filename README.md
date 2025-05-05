

# 🚨 NSL-KDD Intrusion Detection System (IDS) 

This project analyzes the NSL-KDD dataset to build a Machine Learning-based Intrusion Detection System (IDS). It involves data preprocessing, visualization, feature engineering, and classification using advanced models like **Random Forest** and **XGBoost**.

---

## 📁 Project Files

* `nsl_kdd.ipynb` – Main Jupyter notebook containing:

  * Dataset loading and preprocessing
  * Visualizations
  * Feature engineering
  * Model training and evaluation

---

## 📌 Dataset

* **Source**: [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
* An improved version of the KDD’99 dataset for benchmarking network intrusion detection systems.
* Contains 41 features and labels (normal or specific attack types).

---

## 🚀 Workflow Overview

### 🔹 Step 1: Data Loading & Description

* Overview of `KDDTrain+` and `KDDTest+`
* Label distributions and class imbalance

### 🔹 Step 2: Preprocessing

* Label Encoding (e.g., protocol type, service)
* Feature Scaling using StandardScaler

### 🔹 Step 3: Exploratory Data Analysis (EDA)

* Label distribution
* Protocol and service analysis
* Correlation heatmaps for feature relationships

### 🔹 Step 4: Feature Engineering

* Selection of top features using correlation
* Reducing dimensionality while preserving signal

### 🔹 Step 5: Model Building & Evaluation

Trained and evaluated the following ML models:

* 🌲 `RandomForestClassifier`
* ⚡ `XGBClassifier` (Extreme Gradient Boosting)

### 📊 Metrics Used

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

## 📌 Requirements

Install the necessary packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

---

## 🧠 Insights & Observations

* Attack types like `DoS` and `Probe` dominate in terms of frequency.
* Some features like `src_bytes`, `dst_bytes`, and `flag` correlate strongly with attack presence.
* Ensemble models (Random Forest, XGBoost) performed better than linear classifiers.

---

## 📝 How to Run

1. Clone this repository or download the notebook.
2. Download the [NSL-KDD data](https://www.unb.ca/cic/datasets/nsl.html) and place it in your working directory.
3. Launch Jupyter and run the notebook:

```bash
jupyter notebook nsl_kdd.ipynb
```

---

## 📚 References

* Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set.
* [NSL-KDD Dataset Page](https://www.unb.ca/cic/datasets/nsl.html)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)




