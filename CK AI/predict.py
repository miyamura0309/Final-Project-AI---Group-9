import numpy as np
import librosa
import joblib
from pathlib import Path

# ── Cấu hình (phải giống hệt notebook) ────────────────────
SAMPLE_RATE = 16000
DURATION    = 1.0
N_MFCC      = 20
N_FFT       = 512
HOP_LENGTH  = 160
WIN_LENGTH  = 400

# ── Load model ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
model   = joblib.load(BASE_DIR / 'model' / 'svm_model.pkl')
scaler  = joblib.load(BASE_DIR / 'model' / 'scaler.pkl')
le      = joblib.load(BASE_DIR / 'model' / 'label_encoder.pkl')

print("✅ Đã load model thành công!")


def load_and_preprocess(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    return audio

def extract_features(audio):
    mfcc   = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window='hamming'
    )
    delta  = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def stats4(m):
        return np.concatenate([
            np.mean(m, axis=1), np.std(m, axis=1),
            np.min(m,  axis=1), np.max(m, axis=1)
        ])

    def stats2(m):
        return np.concatenate([np.mean(m, axis=1), np.std(m, axis=1)])

    zcr    = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    rms    = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    sp_cen = librosa.feature.spectral_centroid(
                 y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    sp_rol = librosa.feature.spectral_rolloff(
                 y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, roll_percent=0.85)

    extra = np.array([
        float(np.mean(zcr)),    float(np.std(zcr)),
        float(np.mean(rms)),    float(np.std(rms)),
        float(np.mean(sp_cen)), float(np.std(sp_cen)),
        float(np.mean(sp_rol)), float(np.std(sp_rol))
    ])

    return np.concatenate([stats4(mfcc), stats2(delta), stats2(delta2), extra])


def predict(file_path):
    audio     = load_and_preprocess(file_path)
    feature   = extract_features(audio)
    feature_s = scaler.transform([feature])

    pred_idx   = model.predict(feature_s)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    proba      = model.predict_proba(feature_s)[0]

    probabilities = {
        cls: round(float(prob) * 100, 1)
        for cls, prob in zip(le.classes_, proba)
    }

    return {
        "label"        : pred_label,
        "confidence"   : round(float(np.max(proba)) * 100, 1),
        "probabilities": probabilities
    }