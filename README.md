# IsaretlerS-stemler-Final-Faz3


Bu proje, konuşma seslerinden gerçek zamanlı olarak 5 farklı duyguyu (Nötr, Mutlu, Öfkeli, Üzgün, Şaşkın) tespit edebilen bir Makine Öğrenmesi pipeline'ıdır.

## 🚀 Proje Başarısı
* **Score-Board Accuracy:** `%89.00`
* **Kullanılan Mimari:** Soft Voting Classifier (SVM + XGBoost + LightGBM)
* **Öznitelikler (Features):** MFCC (20), Delta, Delta-Delta, Spectral Centroid, Chroma, Tonnetz, RMS, ZCR, Kurtosis, Skewness.

## 🛠️ Kurulum ve Canlı Demo
Proje gerçek zamanlı ses analizi için Streamlit arayüzüne sahiptir. Çalıştırmak için:

1. Gerekli kütüphaneleri yükleyin:
   `pip install -r requirements.txt` *(librosa, scikit-learn, xgboost, lightgbm, streamlit)*
2. Arayüzü başlatın:
   `streamlit run app.py`

## 🧠 Geliştirme Süreci (Faz 3)
Modelin aşırı uyumunu (overfitting) engellemek ve canlı mikrofon testlerindeki "Domain Shift" sorununu aşmak için **Data Augmentation** (Noise Injection & Pitch Shifting) uygulanmış ve veri seti sentetik olarak 3 katına çıkarılmıştır.

**Grup 4:**
* Furkan Yıldız
* Ayberkhan Yüksel
* Recep Çetin
