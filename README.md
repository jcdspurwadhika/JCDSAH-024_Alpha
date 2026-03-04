# JCDSAH-024_Alpha

# 📦 E-Commerce Customer Churn Prediction

> Prediksi churn pelanggan pada platform e-commerce India menggunakan machine learning end-to-end — dari analisis bisnis hingga deployment Streamlit.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://churn-prediction-app-alpha.streamlit.app/)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-1f77b4?logo=tableau)](https://public.tableau.com/app/profile/nisrina.musyaffa.aseno/viz/CustomerChurnRiskAnalysis/Dashboard12?publish=yes)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

---

## 📌 Daftar Isi

1. [Latar Belakang](#-latar-belakang)
2. [Business Problem Statement](#-business-problem-statement)
3. [Dataset](#-dataset)
4. [Alur End-to-End Project](#-alur-end-to-end-project)
5. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
6. [Data Preprocessing](#-data-preprocessing)
7. [Machine Learning Models](#-machine-learning-models)
8. [Hasil Evaluasi](#-hasil-evaluasi)
9. [Analisis Cost-Benefit](#-analisis-cost-benefit)
10. [Business Recommendation](#-business-recommendation)
11. [Tableau Dashboard](#-tableau-dashboard)
12. [Streamlit App & Deployment](#-streamlit-app--deployment)
13. [Tim](#-tim)

---

## 🏢 Latar Belakang

Industri e-commerce di India mengalami pertumbuhan pesat, namun juga sangat kompetitif. **Switching cost yang rendah** menyebabkan pelanggan mudah berpindah platform. Saat ini, churn rate perusahaan berada di angka **16.84%** — artinya sekitar 1 dari 6 pelanggan aktif berhenti menggunakan layanan setiap periode.

Kondisi ini berdampak langsung pada:
- Penurunan pendapatan akibat kehilangan pelanggan aktif
- Tekanan profitabilitas karena **biaya akuisisi pelanggan baru (CAC) 5–7× lebih mahal** dibanding mempertahankan pelanggan yang ada
- Strategi pemasaran yang tidak efisien tanpa sistem identifikasi risiko churn

---

## 🎯 Business Problem Statement

> **Bagaimana mengidentifikasi dan memprediksi pelanggan e-commerce yang berisiko churn berdasarkan karakteristik demografis dan perilaku transaksi mereka?**

### Goals

| # | Goal |
|---|------|
| 1 | Membangun model klasifikasi churn berbasis machine learning |
| 2 | Mengidentifikasi faktor-faktor utama yang memengaruhi keputusan churn |
| 3 | Mengetahui karakteristik demografis & perilaku pelanggan yang berisiko churn |

### Stakeholder

| Stakeholder | Kebutuhan |
|---|---|
| Tim Marketing | Campaign retensi tepat sasaran |
| Tim CRM / Retention | Deteksi dini pelanggan high-risk |
| Tim Customer Service | Prioritas penanganan komplain berisiko tinggi |
| Manajemen / C-Level | Insight strategis untuk optimasi bisnis |
| Tim Data / Analyst | Monitoring pola perilaku & efektivitas retensi |

---

## 📊 Dataset

**Sumber:** [Kaggle – E-Commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

- **5,630 pelanggan** | **20 fitur**
- Target variabel: `Churn` (Binary: 1 = Churn, 0 = Tidak)

### Deskripsi Fitur

| Attribute | Data Type | Description |
|---|---|---|
| CustomerID | Integer | ID unik pelanggan |
| Tenure | Float | Lama pelanggan menggunakan layanan (bulan) |
| PreferredLoginDevice | Object | Perangkat login yang digunakan (Mobile / Computer) |
| CityTier | Integer | Tingkat kota pelanggan (1 = Metro, 3 = Kecil) |
| WarehouseToHome | Float | Jarak warehouse ke rumah pelanggan (km) |
| PreferredPaymentMode | Object | Metode pembayaran yang digunakan |
| Gender | Object | Jenis kelamin pelanggan |
| HourSpendOnApp | Float | Rata-rata waktu yang dihabiskan di aplikasi (jam) |
| NumberOfDeviceRegistered | Integer | Jumlah perangkat yang terdaftar |
| PreferedOrderCat | Object | Kategori produk yang paling sering dibeli |
| SatisfactionScore | Integer | Skor kepuasan layanan (1–5) |
| MaritalStatus | Object | Status pernikahan pelanggan |
| NumberOfAddress | Integer | Jumlah alamat pengiriman yang terdaftar |
| Complain | Binary | Apakah pelanggan pernah mengajukan komplain (1/0) |
| OrderAmountHikeFromLastYear | Float | Persentase kenaikan nilai order dibanding tahun lalu |
| CouponUsed | Float | Jumlah kupon yang telah digunakan |
| OrderCount | Float | Total jumlah transaksi |
| DaySinceLastOrder | Float | Jumlah hari sejak transaksi terakhir |
| CashbackAmount | Float | Total cashback yang diterima pelanggan |
| Churn | Binary | **Target variabel** (1 = Churn, 0 = Tidak Churn) |

---

## 🔄 Alur End-to-End Project

```
1. Business Understanding
      ↓
2. Data Understanding & EDA
      ↓
3. Data Preprocessing
      ↓
4. Baseline Modeling (6 algoritma)
      ↓
5. Imbalance Handling Experiment
      ↓
6. Hyperparameter Tuning (RF, XGBoost, LightGBM)
      ↓
7. Threshold Optimization (OOF-based)
      ↓
8. Final Model Evaluation + SHAP
      ↓
9. Cost-Benefit Analysis
      ↓
10. Business Recommendation
      ↓
11. Deployment (Streamlit + Tableau)
```

---

## 🔍 Exploratory Data Analysis (EDA)

### Insight Numerik

| Variabel | Insight |
|---|---|
| **Tenure** | Pelanggan churn rata-rata hanya 3.4 bulan, vs 11.5 bulan non-churn. Perbedaan signifikan (p < 0.001). |
| **DaySinceLastOrder** | Pelanggan churn justru lebih baru berbelanja (3.2 hari vs 4.8 hari). Bisa mencerminkan perilaku "belanja terakhir". |
| **CashbackAmount** | Non-churn menerima cashback lebih tinggi (₹180 vs ₹160). Cashback berfungsi sebagai alat retensi. |
| **CouponUsed** | Tidak signifikan (p = 0.54) — penggunaan kupon tidak cukup untuk membedakan churn. |
| **WarehouseToHome** | Pelanggan churn sedikit lebih jauh dari gudang (~17 km vs ~15 km). |

### Insight Kategorikal

| Variabel | Insight |
|---|---|
| **Complain** | Pelanggan dengan riwayat komplain memiliki churn rate **31.7%** vs hanya 10.9%. Sinyal paling kuat. |
| **MaritalStatus** | Pelanggan Single churn **26.7%**, sementara Married hanya **11.5%**. |
| **PreferredPaymentMode** | COD memiliki churn t **24.9%**, jauh lebih tinggi dari kartu digital (~14%). |
| **PreferedOrderCat** | Kategori Mobile Phone memiliki churn tertinggi **(~27.4%)**. |
| **CityTier** | Kota Tier 3 lebih berisiko churn (21.4%) dibanding Tier 1 (14.5%). |

### Segmen Risiko Tertinggi
> **Single + COD + pernah komplain + beli Mobile Phone + CityTier 3**

---

## ⚙️ Data Preprocessing

### Penanganan Missing Value

Terdapat missing value pada 7 kolom numerik (proporsi 4.5–5.5%). Mekanisme dianalisis dengan:

1. **Chi-Square Test** — 6 dari 7 kolom berkorelasi signifikan dengan target → indikasi **MAR (Missing At Random)**
2. **Missing Indicator Model** — pola missing dapat diprediksi dari fitur lain (AUC ≥ 0.6)
3. **Masking Validation** — Iterative Imputer memiliki MAE lebih rendah dibanding Median Imputer

**Keputusan:** Imputasi menggunakan `IterativeImputer` yang memanfaatkan hubungan multivariat antar fitur. *Fit hanya pada data training* untuk mencegah data leakage.

### Data Cleaning

- Merge kategori duplikat: `Phone` & `Mobile Phone`, `COD` & `Cash on Delivery`, `CC` & `Credit Card`
- Drop kolom `CustomerID` (tidak informatif untuk model)
- Tidak ada data duplikat

### Pipeline Preprocessing

```python
num_pipeline = Pipeline([
    ('imputer', IterativeImputer(random_state=42)),
    ('scaler',  StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
```

- **Train:Test Split** = 80:20, stratified berdasarkan target
- **StandardScaler** hanya digunakan untuk model yang sensitif terhadap skala (Logistic Regression, KNN)
- Model tree-based tidak memerlukan scaling

---

## 🤖 Machine Learning Models

### Pendekatan

**Supervised Learning – Binary Classification** dengan membandingkan 6 algoritma:

| Model | Tipe | Keterangan |
|---|---|---|
| Logistic Regression | Linear | Baseline interpretable |
| K-Nearest Neighbors | Distance-based | Menangkap pola lokal |
| Decision Tree | Tree-based | Aturan keputusan non-linear |
| Random Forest | Ensemble (Bagging) | Stabil, mengurangi overfitting |
| XGBoost | Ensemble (Boosting) | Sequential tree boosting |
| LightGBM | Ensemble (Boosting) | Efisien, scalable |

### Metrik Evaluasi

| Metrik | Peran |
|---|---|
| **Recall** ⭐ *Primary* | Meminimalkan False Negative (FN) |
| **Precision** | Mengontrol False Positive (FP) |
| **F1-Score** | Trade-off Precision–Recall |
| **PR-AUC** | Metrik utama untuk dataset imbalanced |
| **ROC-AUC** | Kemampuan separasi kelas secara umum |
| **Cross-Validation** | Memastikan stabilitas dan mencegah overfitting |

> **Alasan Recall diprioritaskan:** False Negative (pelanggan churn yang tidak terdeteksi) jauh lebih mahal daripada False Positive. FN → kehilangan pelanggan + bayar CAC baru. FP → hanya biaya retensi yang relatif kecil.

### Eksperimen Imbalance Handling

Empat teknik dievaluasi pada Random Forest sebagai model representatif:

| Teknik | Pendekatan |
|---|---|
| Baseline | Tanpa penanganan |
| `class_weight='balanced'` | Cost-sensitive learning |
| SMOTE | Sintesis sampel minority |
| ROS (Random OverSampling) | Duplikasi sampel minority |
| RUS (Random UnderSampling) | Pengurangan sampel majority |

**Keputusan:** Tidak menggunakan resampling pada model final karena:
- Baseline RF sudah menghasilkan PR-AUC > 0.98 dan Recall > 0.89
- Resampling meningkatkan Recall tetapi mengorbankan Precision signifikan
- **Threshold tuning** lebih fleksibel, reversible, dan tidak menambah noise sintetis

---

## 📈 Hasil Evaluasi

### Cross-Validation Benchmark (5-Fold)

| Model | CV PR-AUC | CV ROC-AUC | CV Recall |
|---|---|---|---|
| **Random Forest** | **0.9300** | 0.9800 | Tinggi |
| XGBoost | 0.9100+ | 0.9750+ | Tinggi |
| LightGBM | 0.9100+ | 0.9750+ | Tinggi |
| Logistic Regression | ~0.72 | ~0.89 | Sedang |
| KNN | ~0.76 | ~0.88 | Sedang |
| Decision Tree | Terendah | Terendah | Rendah |

> KNN terindikasi **overfitting** (Gap Recall = 0.17, Gap F1 = 0.13).

### Hasil Hyperparameter Tuning (RF, XGBoost, LightGBM)

Tuning menggunakan `RandomizedSearchCV` (5-fold, `scoring='recall'`).

### 🏆 Model Final: LightGBM

Setelah tuning dan optimasi threshold (dipilih dari OOF predictions berdasarkan F1 optimal):

| Metrik | Nilai |
|---|---|
| **ROC-AUC** | **≈ 1.000** |
| **PR-AUC** | **≈ 0.998** |
| **Recall (Churn)** | **0.97** |
| **Precision (Churn)** | **0.98** |
| **F1-Score (Churn)** | **0.98** |
| **Accuracy** | **99%** |
| False Negative (FN) | **6** |
| False Positive (FP) | **3** |

**Stabilitas (10-Fold CV):** Val Recall rata-rata 0.885, ROC-AUC rata-rata 0.987 — konsisten di seluruh fold.

### SHAP Feature Importance

Fitur paling berpengaruh terhadap churn (berurutan):

1. **Tenure** — Pelanggan baru (tenure rendah) sangat berisiko churn
2. **Complain** — Riwayat komplain secara drastis menaikkan risiko
3. **DaySinceLastOrder** — Lama tidak bertransaksi meningkatkan risiko
4. **CashbackAmount** — Cashback rendah berkorelasi dengan churn lebih tinggi
5. **WarehouseToHome** — Jarak pengiriman yang lebih jauh meningkatkan risiko

---

## 💰 Analisis Cost-Benefit

**Asumsi biaya:**
- Customer Retention Cost (CRC): **₹300** per pelanggan
- Customer Acquisition Cost (CAC): **₹1,500** per pelanggan baru

| Skenario | Deskripsi | Total Biaya |
|---|---|---|
| **No Action** | Semua pelanggan churn tidak ditangani | Tertinggi |
| **Retensi Massal** | Semua pelanggan diberi promo tanpa seleksi | Sangat tinggi |
| **Targeted (dengan model)** | Hanya pelanggan berisiko yang diberi intervensi | **Terendah** |

> Model berhasil **menghemat biaya signifikan** dibandingkan kedua skenario alternatif, karena FN dan FP yang sangat kecil.

---

## 💡 Business Recommendation

### 1. Program Onboarding Intensif di Awal Masa Pelanggan
Tenure adalah faktor churn terkuat. Pelanggan churn rata-rata baru bergabung 3–4 bulan. Buat program onboarding terstruktur: voucher bertahap, panduan fitur, check-in aktif di bulan ke-1 dan ke-3.

### 2. SLA Resolusi Komplain
Pelanggan yang pernah komplain memiliki churn rate 3× lebih tinggi. Tetapkan SLA internal untuk resolusi komplain dan tandai pelanggan berisiko di CRM secara otomatis.

### 3. Segmentasi dan Intervensi Berbasis Risiko

| Segmen | Churn Rate | Saran |
|---|---|---|
| Status Single | 26.7% | Program loyalitas personal |
| Pembayaran COD | 24.9% | Insentif migrasi ke payment digital |
| Mobile Phone category | ~27.4% | Tingkatkan pengalaman after-sales |
| CityTier 3 | 21.4% | Tingkatkan kualitas layanan pengiriman |

### 4. Optimalkan Cashback, Bukan Hanya Kupon
Cashback terbukti berpengaruh terhadap retensi, sementara kupon tidak. Pertimbangkan mengalokasikan sebagian budget promo dari diskon satu kali ke cashback yang konsisten dan berkelanjutan.

---

## 📊 Tableau Dashboard

Visualisasi interaktif pola churn pelanggan:

🔗 [Customer Churn Risk Analysis – Tableau Public](https://public.tableau.com/app/profile/nisrina.musyaffa.aseno/viz/CustomerChurnRiskAnalysis/Dashboard12?publish=yes)

---

## 🚀 Streamlit App & Deployment

Aplikasi prediksi churn real-time dapat diakses di:

🔗 **[churn-prediction-app-alpha.streamlit.app](https://churn-prediction-app-alpha.streamlit.app/)**

### Fitur Aplikasi

- 📋 Input data pelanggan secara manual
- 📊 Prediksi probabilitas churn secara real-time
- 🎯 Segmentasi risiko churn (Low / Medium / High)

### Cara Menjalankan Secara Lokal

```bash
# 1. Clone repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan aplikasi Streamlit
streamlit run app.py
```

### Dependencies Utama

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
matplotlib
seaborn
streamlit
scipy
statsmodels
```

---

## 👥 Tim

**JCDSAH-024 – Alpha Team**

Project ini dikembangkan sebagai bagian dari program **Job Connector Data Science** batch 024.

---