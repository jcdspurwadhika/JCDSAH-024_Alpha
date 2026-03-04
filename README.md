# JCDSAH-024_Alpha

Customer Churn Prediction
E-Commerce Customer Churn Analysis and Prediction

Project Customer Churn Prediction merupakan project machine learning yang bertujuan untuk memprediksi apakah seorang pelanggan berpotensi churn (berhenti menggunakan layanan) atau tidak.

Model machine learning dibangun menggunakan data perilaku pelanggan seperti aktivitas transaksi, penggunaan aplikasi, kepuasan pelanggan, dan interaksi dengan layanan.

Goal utama project ini adalah membantu perusahaan mengidentifikasi pelanggan berisiko churn lebih awal, sehingga perusahaan dapat melakukan strategi customer retention yang lebih efektif.

Dataset

Dataset yang digunakan dalam project ini dapat diakses melalui link berikut:

https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

Dataset berisi 5630 pelanggan dengan 20 fitur yang menggambarkan karakteristik demografis dan perilaku pelanggan.

Dataset Attributes
Attribute	Data Type	Description
CustomerID	Integer	ID unik pelanggan
Tenure	Float	Lama pelanggan menggunakan layanan
PreferredLoginDevice	Object	Perangkat yang digunakan untuk login
CityTier	Integer	Tingkat kota pelanggan
WarehouseToHome	Float	Jarak warehouse ke rumah pelanggan
PreferredPaymentMode	Object	Metode pembayaran
Gender	Object	Jenis kelamin pelanggan
HourSpendOnApp	Float	Waktu yang dihabiskan di aplikasi
NumberOfDeviceRegistered	Integer	Jumlah perangkat yang terdaftar
PreferedOrderCat	Object	Kategori produk yang sering dibeli
SatisfactionScore	Integer	Skor kepuasan pelanggan
MaritalStatus	Object	Status pernikahan
NumberOfAddress	Integer	Jumlah alamat pelanggan
Complain	Binary	Apakah pelanggan pernah complain
OrderAmountHikeFromLastYear	Float	Persentase peningkatan nilai order
CouponUsed	Float	Jumlah kupon yang digunakan
OrderCount	Float	Jumlah transaksi
DaySinceLastOrder	Float	Hari sejak transaksi terakhir
CashbackAmount	Float	Jumlah cashback
Churn	Binary	Target variable
Exploratory Data Analysis

Beberapa insight utama dari EDA:

Pelanggan dengan tenure rendah lebih rentan churn

Pelanggan yang pernah complain memiliki churn rate lebih tinggi

Pelanggan yang menerima cashback lebih besar cenderung lebih loyal

Segmen Single + COD + pernah complain memiliki risiko churn tertinggi

Machine Learning Models

Beberapa model yang digunakan dalam project ini:

Logistic Regression

K-Nearest Neighbors

Decision Tree

Random Forest

XGBoost

LightGBM

Evaluasi model menggunakan:

Recall

Precision

F1 Score

ROC-AUC

PR-AUC

Recall diprioritaskan karena False Negative (pelanggan churn yang tidak terdeteksi) lebih mahal dibanding False Positive.

Tableau Dashboard

Dashboard analisis churn pelanggan dapat diakses melalui link berikut:

(isi kalau ada dashboard tableau)

Streamlit App

Aplikasi untuk memprediksi churn pelanggan dapat diakses melalui link berikut:

https://churn-prediction-app-alpha.streamlit.app/

Fitur aplikasi:

Input data pelanggan

Prediksi churn probability

Segmentasi risiko churn

Streamlit Deployment

Model machine learning yang telah dilatih disimpan dalam bentuk file model, kemudian digunakan oleh aplikasi Streamlit untuk melakukan prediksi.

Cara menjalankan Streamlit di localhost
