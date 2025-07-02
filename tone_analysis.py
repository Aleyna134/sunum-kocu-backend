import librosa
import numpy as np

def analyze_tone(audio_path):
    try:
        y, sr = librosa.load(audio_path)

        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

        # Energy
        energy = np.mean(librosa.feature.rms(y=y))

        # Zero Crossing Rate (ZCR)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Geri Bildirimler
        feedback = []

        # Ses tonu değerlendirmesi
        if avg_pitch < 100:
            feedback.append("Ses tonunuz biraz düşük kalmış, daha canlı bir tonlama sunumunuzu güçlendirebilir.")
        elif avg_pitch > 300:
            feedback.append("Ses tonunuz oldukça yüksek; daha doğal bir aralıkta kalmaya çalışın.")
        else:
            feedback.append("Ses tonunuz dengeli ve anlaşılırdı.")

        # Enerji değerlendirmesi
        if energy < 0.01:
            feedback.append("Daha enerjik ve canlı bir konuşma tarzı dinleyiciyi daha çok etkileyebilir.")
        elif energy > 0.05:
            feedback.append("Sesiniz enerjikti, bu da anlatımı destekledi.")
        else:
            feedback.append("Enerji seviyeniz sunuma uygundu.")

        # ZCR değerlendirmesi (tonlama çeşitliliği)
        if zcr < 0.03:
            feedback.append("Konuşma tonlamanız zaman zaman tekdüze olmuş olabilir.")
        elif zcr > 0.1:
            feedback.append("Tonlamanız dinamikti ve dikkat çekiciydi.")
        else:
            feedback.append("Tonlamanız yeterince değişkenlik gösteriyordu.")

        return {
            "summary": " ".join(feedback[:1]),  # Genel yorum (ilk cümle)
            "suggestions": feedback[1:]          # Geri kalanları öneri olarak ver
        }

    except Exception as e:
        return {
            "summary": "Ses tonu analizi sırasında bir hata oluştu.",
            "suggestions": [f"Hata detayı: {str(e)}"]
        }
