# JCDSAH-024_Alpha

# Customer Churn Prediction

## E-Commerce Customer Churn Analysis and Prediction

Project **Customer Churn Prediction** merupakan project **machine learning** yang bertujuan untuk memprediksi apakah seorang pelanggan berpotensi **churn (berhenti menggunakan layanan)** atau tidak.

Model machine learning dibangun menggunakan data perilaku pelanggan seperti **aktivitas transaksi, penggunaan aplikasi, kepuasan pelanggan, serta interaksi dengan layanan**.

Goal utama project ini adalah membantu platform e-commerce di India **mengidentifikasi pelanggan berisiko churn lebih awal**, sehingga perusahaan dapat melakukan strategi **customer retention yang lebih efektif dan terukur**.

---

# Dataset

Dataset yang digunakan dalam project ini dapat diakses melalui link berikut:

Dataset:  
https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

Dataset berisi:

- **5630 pelanggan**
- **20 fitur**

Dataset ini menggambarkan **karakteristik demografis serta perilaku transaksi pelanggan pada platform e-commerce**.

---

# Dataset Attributes

| Attribute | Data Type | Description |
|---|---|---|
| CustomerID | Integer | ID unik pelanggan |
| Tenure | Float | Lama pelanggan menggunakan layanan |
| PreferredLoginDevice | Object | Perangkat yang digunakan untuk login |
| CityTier | Integer | Tingkat kota pelanggan |
| WarehouseToHome | Float | Jarak warehouse ke rumah pelanggan |
| PreferredPaymentMode | Object | Metode pembayaran |
| Gender | Object | Jenis kelamin pelanggan |
| HourSpendOnApp | Float | Waktu yang dihabiskan di aplikasi |
| NumberOfDeviceRegistered | Integer | Jumlah perangkat yang terdaftar |
| PreferedOrderCat | Object | Kategori produk yang sering dibeli |
| SatisfactionScore | Integer | Skor kepuasan pelanggan |
| MaritalStatus | Object | Status pernikahan |
| NumberOfAddress | Integer | Jumlah alamat pelanggan |
| Complain | Binary | Apakah pelanggan pernah complain |
| OrderAmountHikeFromLastYear | Float | Persentase peningkatan nilai order |
| CouponUsed | Float | Jumlah kupon yang digunakan |
| OrderCount | Float | Jumlah transaksi |
| DaySinceLastOrder | Float | Hari sejak transaksi terakhir |
| CashbackAmount | Float | Jumlah cashback |
| Churn | Binary | Target variable |

---

# Exploratory Data Analysis

Beberapa insight utama dari **EDA**:

- Pelanggan dengan **tenure rendah** memiliki kemungkinan churn lebih tinggi  
- Pelanggan yang **pernah melakukan complain** memiliki churn rate jauh lebih tinggi  
- Pelanggan yang menerima **cashback lebih tinggi cenderung lebih loyal**  
- Segmen **Single + COD + pernah complain** memiliki **risiko churn tertinggi**

---

# Machine Learning Models

Beberapa model yang digunakan dalam project ini:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

Evaluasi model menggunakan metrik berikut:

- Recall
- Precision
- F1 Score
- ROC-AUC
- PR-AUC

Metric **Recall** diprioritaskan karena **False Negative (pelanggan churn yang tidak terdeteksi) memiliki dampak bisnis lebih besar dibanding False Positive**.

---

# Tableau Dashboard


# Streamlit App

Aplikasi untuk memprediksi churn pelanggan dapat diakses melalui link berikut:

https://churn-prediction-app-alpha.streamlit.app/

Fitur aplikasi:

- Input data pelanggan
- Prediksi churn probability
- Segmentasi risiko churn pelanggan

---

# Streamlit Deployment

Model machine learning yang telah dilatih disimpan dalam bentuk file model dan digunakan oleh aplikasi **Streamlit** untuk melakukan prediksi secara real-time.

### Cara menjalankan Streamlit di localhost

1. Install library yang diperlukan

Segmentasi risiko churn pelanggan

Streamlit Deployment

Model machine learning yang telah dilatih disimpan dalam bentuk file model dan digunakan oleh aplikasi Streamlit untuk melakukan prediksi secara real-time.
