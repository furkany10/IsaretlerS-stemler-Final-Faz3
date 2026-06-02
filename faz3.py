"""
BIL216 - Emo-Challenge 2026
FAZ 3: DATA AUGMENTATION (VERİ ÇOĞALTMA) & ENSEMBLE
Grup 05 - Hedef: 90%+ Doğruluk
"""
import joblib
import os
import numpy as np
import librosa
import scipy.stats as stats
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================
VERI_KLASORU = r"C:\Users\omenh\OneDrive\Desktop\Midterm_Dataset_2026" 
# =============================================

DUYGU_ETIKETLERI = ['Notr', 'Mutlu', 'Ofkeli', 'Uzgun', 'Saskin']

def etiket_cikar(dosya_adi):
    for duygu in DUYGU_ETIKETLERI:
        if duygu in dosya_adi: return duygu
    return None

# Fonksiyonu dosya yolundan değil, doğrudan ses verisi (y) üzerinden çalışacak şekilde güncelledik
def ozellik_cikar_dizi(y, sr):
    try:
        if len(y) == 0: return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfcc_features = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1)
        ])

        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        basic_features = np.array([
            np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms),
            np.mean(spec_cent), np.std(spec_cent),
            np.mean(spec_roll), np.std(spec_roll)
        ])
        chroma_features = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        new_spectral_features = np.concatenate([
            np.mean(spec_contrast, axis=1), np.std(spec_contrast, axis=1),
            np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)
        ])

        signal_skew = stats.skew(y)
        signal_kurtosis = stats.kurtosis(y)
        statistical_features = np.array([signal_skew, signal_kurtosis])

        return np.concatenate([mfcc_features, basic_features, chroma_features, new_spectral_features, statistical_features])
    except:
        return None

def veri_yukle_ve_cogalt(ana_klasor):
    X, y_labels = [], []
    alt_klasorler = [os.path.join(ana_klasor, d) for d in os.listdir(ana_klasor) 
                     if os.path.isdir(os.path.join(ana_klasor, d)) and (d.startswith('GROUP') or d.startswith('GRUP'))]
    
    tum_dosyalar = []
    for klasor in alt_klasorler:
        for dosya in os.listdir(klasor):
            if dosya.lower().endswith('.wav'): tum_dosyalar.append(os.path.join(klasor, dosya))

    print(f"Toplam {len(tum_dosyalar)} orijinal dosya var. Veri çoğaltma (Augmentation) ile bu sayı 3 katına çıkacak...")
    print("⏳ Öznitelik çıkarma işlemi başladı, bu biraz uzun sürebilir. Lütfen bekleyin...")
    
    for dosya_yolu in tum_dosyalar:
        etiket = etiket_cikar(os.path.basename(dosya_yolu))
        if etiket is None: continue
        
        try:
            # Sesi bir kez yükle
            y_orj, sr = librosa.load(dosya_yolu, duration=5, sr=22050)
            
            # 1. ORİJİNAL SES
            ozellik_orj = ozellik_cikar_dizi(y_orj, sr)
            if ozellik_orj is not None:
                X.append(ozellik_orj)
                y_labels.append(etiket)

            # 2. GÜRÜLTÜ EKLENMİŞ SES (Noise Injection)
            noise = np.random.randn(len(y_orj))
            y_noise = y_orj + 0.005 * noise
            ozellik_noise = ozellik_cikar_dizi(y_noise, sr)
            if ozellik_noise is not None:
                X.append(ozellik_noise)
                y_labels.append(etiket)

            # 3. PERDESİ KAYDIRILMIŞ SES (Pitch Shifting)
            # Sesi 2 adım inceltiyoruz
            y_pitch = librosa.effects.pitch_shift(y=y_orj, sr=sr, n_steps=2)
            ozellik_pitch = ozellik_cikar_dizi(y_pitch, sr)
            if ozellik_pitch is not None:
                X.append(ozellik_pitch)
                y_labels.append(etiket)
                
        except Exception as e:
            continue
            
    return np.array(X), np.array(y_labels)

if __name__ == "__main__":
    X, y = veri_yukle_ve_cogalt(VERI_KLASORU)
    print(f"✅ Augmentation Tamamlandı! Yeni veri sayımız: {len(X)}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, "le.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    print("\nModeller Eğitiliyor...")
    # Veri setimiz artık daha büyük olduğu için modellere daha fazla güvenebiliriz
    svm_best = SVC(C=50, gamma=0.001, kernel='rbf', probability=True, random_state=42)
    xgb_best = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='mlogloss')
    lgb_best = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=-1)

    print("Voting Classifier birleştiriliyor...")
    voting_clf = VotingClassifier(
        estimators=[('svm', svm_best), ('xgb', xgb_best), ('lgb', lgb_best)],
        voting='soft'
    )
    
    voting_clf.fit(X_train_scaled, y_train)
    final_pred = voting_clf.predict(X_test_scaled)
    final_acc = accuracy_score(y_test, final_pred)

    joblib.dump(voting_clf, "FinalProje_Model_GRUP05.pkl")

    print("\n" + "=" * 60)
    print(f"🔥 DATA AUGMENTATION NİHAİ SCORE-BOARD ACCURACY: %{final_acc * 100:.2f}")
    print("=" * 60)

    cm = confusion_matrix(y_test, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Faz 3 Data Augmentation - Karışıklık Matrisi')
    plt.tight_layout()
    plt.savefig('FinalProje_ConfusionMatrix_GRUP05.png', dpi=150)
    print("✅ Model, Scaler, Encoder ve Grafik başarıyla kaydedildi.")