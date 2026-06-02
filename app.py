"""
BIL216 - Emo-Challenge 2026
Canlı Demo ve Web Arayüzü (Gürültü Filtreli ve Normalize Sürüm)
Grup 05
"""
import streamlit as st
import librosa
import numpy as np
import joblib
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Sayfa Ayarları
st.set_page_config(page_title="Emo-Challenge 2026 - Canlı Demo", page_icon="🎙️", layout="centered")

st.title("🎙️ Duygu Tanıma Canlı Demo - Grup 05")
st.markdown("**Faz 3:** Optimize edilmiş Ensemble modelimiz ve *Canlı Ses İyileştirme Filtreleri* kullanılarak gerçek zamanlı duygu analizi yapılmaktadır.")

# Model, Scaler ve LabelEncoder'ı Yükle
@st.cache_resource
def load_system():
    try:
        model = joblib.load("FinalProje_Model_GRUP05.pkl") 
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("le.pkl")
        return model, scaler, le
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu. Lütfen eğitim kodunu çalıştırdığınızdan emin olun. Hata: {e}")
        return None, None, None

model, scaler, le = load_system()

# Canlı veri için öznitelik çıkarma fonksiyonu
def extract_live_features(y, sr):
    try:
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

        features = np.concatenate([mfcc_features, basic_features, chroma_features, new_spectral_features, statistical_features])
        return features.reshape(1, -1) 
    except Exception as e:
        st.error(f"Öznitelik çıkarılırken hata: {e}")
        return None

# Sekmeler oluştur
tab1, tab2 = st.tabs(["🎙️ Mikrofondan Analiz", "📁 Dosyadan Analiz"])

with tab1:
    st.info("Lütfen mikrofona konuşun ve Analiz Et butonuna basın.")
    audio_value = st.audio_input("Sesinizi kaydedin:")
    
    if audio_value is not None:
        if st.button("🎤 Kaydı Analiz Et"):
            with st.spinner("Analiz ediliyor..."):
                try:
                    # FİLTRE YOK! Sadece eğitimdeki gibi ilk 5 saniyeyi ham haliyle alıyoruz.
                    y, sr = librosa.load(audio_value, duration=5, sr=22050)
                    
                    features = extract_live_features(y, sr)
                    
                    if features is not None and model is not None:
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)
                        emotion = le.inverse_transform(prediction)[0]
                        
                        if emotion.lower() == 'mutlu':
                            st.success(f"😊 **Tespit Edilen Duygu:** {emotion.upper()}")
                        elif emotion.lower() == 'ofkeli':
                            st.error(f"😠 **Tespit Edilen Duygu:** {emotion.upper()}")
                        elif emotion.lower() == 'uzgun':
                            st.info(f"😢 **Tespit Edilen Duygu:** {emotion.upper()}")
                        elif emotion.lower() == 'saskin':
                            st.warning(f"😲 **Tespit Edilen Duygu:** {emotion.upper()}")
                        else:
                            st.success(f"😐 **Tespit Edilen Duygu:** {emotion.upper()}")
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")

with tab2:
    uploaded_file = st.file_uploader("Bir WAV ses dosyası yükleyin", type=['wav'])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("📁 Dosyayı Analiz Et"):
            with st.spinner("Analiz ediliyor..."):
                try:
                    # FİLTRE YOK! 
                    y, sr = librosa.load(uploaded_file, duration=5, sr=22050)
                    
                    features = extract_live_features(y, sr)
                    
                    if features is not None and model is not None:
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)
                        emotion = le.inverse_transform(prediction)[0]
                        
                        st.success(f"**Tespit Edilen Duygu:** {emotion.upper()}")
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")