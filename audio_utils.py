from moviepy.editor import VideoFileClip
import librosa
import numpy as np

def extract_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)

def detect_filler_like_segments(audio_path, threshold_db=-35, min_duration=0.3):
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    low_energy = rms_db < threshold_db
    filler_segments = []
    current_start = None

    for i, is_low in enumerate(low_energy):
        t = i * hop_length / sr
        if is_low:
            if current_start is None:
                current_start = t
        else:
            if current_start is not None:
                duration = t - current_start
                if duration >= min_duration:
                    filler_segments.append((current_start, t))
                current_start = None

    return filler_segments
