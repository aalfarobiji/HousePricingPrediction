# 🏠 House Price Prediction

Proyek machine learning untuk memprediksi harga rumah menggunakan dataset [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). Notebook ini membandingkan tiga model regresi dan melakukan hyperparameter tuning untuk mendapatkan performa terbaik.

---

## 📋 Daftar Isi

- [Overview](#overview)
- [Dataset](#dataset)
- [Struktur Notebook](#struktur-notebook)
- [Feature Engineering](#feature-engineering)
- [Model yang Digunakan](#model-yang-digunakan)
- [Evaluasi Model](#evaluasi-model)
- [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
- [Dependensi](#dependensi)

---

## Overview

Proyek ini bertujuan memprediksi harga jual rumah (`SalePrice`) berdasarkan berbagai fitur seperti luas bangunan, kondisi rumah, lokasi, dan lainnya. Pipeline mencakup eksplorasi data, rekayasa fitur, preprocessing, pelatihan model, hyperparameter tuning, cross-validation, hingga analisis feature importance.

---

## Dataset

Dataset yang digunakan adalah **Ames Housing Dataset** dari Kaggle.

| File | Keterangan |
|------|------------|
| `train.csv` | Data latih dengan label `SalePrice` |

> 📥 Download dataset dari: [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

---

## Struktur Notebook

```
House_Prediction_V2.ipynb
│
├── A. Import Library
├── B. Exploratory Data Analysis (EDA)
├── C. Data Preprocessing
│   ├── Feature Engineering
│   ├── Feature Dropping
│   ├── Train-Test Split
│   └── Preprocessing Pipeline
├── D. Data Modelling
│   ├── Linear Regression
│   ├── Random Forest
│   └── XGBoost
├── E. Model Evaluation
├── F. Hyperparameter Tuning
│   ├── Random Forest Tuning (GridSearchCV)
│   └── XGBoost Tuning (GridSearchCV)
├── G. Evaluate Best Model
├── H. Cross Validation
├── I. Feature Importance
└── J. Additional Test (Model Comparison Table)
```

---

## Feature Engineering

Beberapa fitur baru dibuat untuk meningkatkan performa model:

| Fitur Baru | Formula | Keterangan |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Total luas bangunan |
| `HouseAge` | `YrSold - YearBuilt` | Usia rumah saat dijual |
| `RemodAge` | `YrSold - YearRemodAdd` | Usia sejak renovasi terakhir |
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | Total kamar mandi |
| `BedroomRatio` | `BedroomAbvGr / (TotRmsAbvGrd + 1)` | Rasio kamar tidur terhadap total ruangan |

Fitur-fitur original yang sudah digabungkan kemudian di-drop untuk menghindari redundansi.

---

## Model yang Digunakan

### 1. Linear Regression
Model baseline menggunakan regresi linier standar dari Scikit-learn.

### 2. Random Forest Regressor
Model ensemble berbasis decision tree dengan hyperparameter tuning:
```python
param_grid_rf = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None]
}
```

### 3. XGBoost Regressor
Model gradient boosting dengan tuning lebih komprehensif:
```python
param_grid_xgb = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 6],
    'model__learning_rate': [0.01, 0.1],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}
```

### Preprocessing Pipeline

```
Numerical Features  →  Median Imputer  →  StandardScaler  ─┐
                                                             ├──▶  ColumnTransformer  ──▶  Model
Categorical Features →  Constant Imputer → OneHotEncoder   ─┘
```

---

## Evaluasi Model

Model dievaluasi menggunakan beberapa metrik:

| Metrik | Keterangan |
|--------|------------|
| **RMSE** | Root Mean Squared Error (skala asli harga) |
| **MAE** | Mean Absolute Error |
| **R²** | Koefisien determinasi |
| **MAPE** | Mean Absolute Percentage Error (Train & Test) |
| **Consistency Gap** | Selisih MAPE Train vs Test (indikator overfitting) |

**Kriteria Stabilitas Model:**

| Gap (Train vs Test MAPE) | Status |
|--------------------------|--------|
| < 2% | Sangat Stabil ✅ |
| 2% – 5% | Cukup Stabil ⚠️ |
| > 5% | Overfitting ❌ |

Cross-validation dilakukan dengan **5-fold CV** menggunakan scoring `neg_root_mean_squared_error`.

> Target variabel (`SalePrice`) di-transformasi menggunakan **log1p** untuk menormalkan distribusi, dan dikembalikan ke skala asli menggunakan **expm1** saat evaluasi.

---

## Instalasi & Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/house-price-prediction.git
cd house-price-prediction
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download `train.csv` dari [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dan letakkan di direktori yang sama dengan notebook.

### 4. Jalankan Notebook

```bash
jupyter notebook House_Prediction_V2.ipynb
```

---

## Dependensi

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

Atau install sekaligus:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

---

## 📁 Struktur Direktori

```
house-price-prediction/
├── House_Prediction_V2.ipynb   # Notebook utama
├── train.csv                   # Dataset (download dari Kaggle)
├── requirements.txt            # Daftar dependensi
└── README.md                   # Dokumentasi proyek
```

---

## 📌 Catatan

- Notebook menggunakan `random_state=42` untuk reproducibility.
- Hyperparameter tuning menggunakan `GridSearchCV` dengan `n_jobs=-1` (memanfaatkan semua core CPU).
- Model terbaik dipilih berdasarkan kombinasi Test RMSE terendah dan Consistency Gap terkecil.
