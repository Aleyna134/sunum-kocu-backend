from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import subprocess
from pdf2image import convert_from_path
from moviepy.editor import VideoFileClip
import whisper
from openai import OpenAI
import traceback
from dotenv import load_dotenv
import re

from gaze_tracker import analyze_gaze
from audio_utils import detect_filler_like_segments
from tone_analysis import analyze_tone
from posture_analysis import analyze_posture 


# Ortam değişkenlerini yükle
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
CONVERTED_DIR = "converted"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONVERTED_DIR, exist_ok=True)

@app.get("/")
async def read_root():
    return {"message": "Sunum Koçu API çalışıyor."}

@app.post("/upload/")
async def upload_presentation(file: UploadFile = File(...)):
    try:
        filename = os.path.basename(file.filename.lower())
        if len(filename) > 100:
            filename = filename[-100:]

        uid = str(uuid.uuid4())
        save_folder = os.path.join(CONVERTED_DIR, uid)
        os.makedirs(save_folder, exist_ok=True)

        upload_path = os.path.join(UPLOAD_DIR, f"{uid}_{filename}")
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        if filename.endswith(".pptx"):
            libreoffice_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            subprocess.run([
                libreoffice_path,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", save_folder,
                upload_path
            ], check=True)
            converted_pdf = next((os.path.join(save_folder, f)
                                  for f in os.listdir(save_folder) if f.endswith(".pdf")), None)
            if not converted_pdf:
                return JSONResponse(status_code=500, content={"error": "PDF'e dönüştürülemedi."})
            images = convert_from_path(converted_pdf, dpi=150)
        elif filename.endswith(".pdf"):
            images = convert_from_path(upload_path, dpi=150)
        else:
            return JSONResponse(status_code=400, content={"error": "Sadece .pptx ve .pdf dosyaları destekleniyor."})

        image_urls = []
        for i, image in enumerate(images):
            img_path = os.path.join(save_folder, f"slide_{i + 1}.png")
            image.save(img_path, "PNG")
            image_urls.append(f"http://localhost:8000/{CONVERTED_DIR}/{uid}/slide_{i + 1}.png")

        return {"image_urls": image_urls}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Sunum yükleme hatası: {str(e)}"})

@app.post("/upload_video/")
async def upload_video(video: UploadFile = File(...)):
    try:
        uid = str(uuid.uuid4())
        ext = video.filename.split('.')[-1]
        video_path = os.path.join(UPLOAD_DIR, f"{uid}.{ext}")

        with open(video_path, "wb") as f:
            f.write(await video.read())

        fixed_video_path = os.path.join(UPLOAD_DIR, f"{uid}_fixed.mp4")
        subprocess.run([
            "ffmpeg", "-i", video_path, "-c:v", "copy", "-c:a", "aac", fixed_video_path
        ], check=True)

        # Sesi çıkar
        audio_path = os.path.join(UPLOAD_DIR, f"{uid}.wav")
        clip = VideoFileClip(fixed_video_path)
        clip.audio.write_audiofile(audio_path)
        clip.close()

        # Transkripsiyon
        model = whisper.load_model("large-v3")
        result = model.transcribe(audio_path)
        transcript = result["text"]

        # Dolgu kelimeler
        patterns = {
            "ııı": r"\b(ı{1,3}|i{1,3}|uh+|umm+|hı+h+)\b",
            "eee": r"\b(e{2,}|eh+|em+|eee+)\b",
            "şey": r"\bşey\b",
            "yani": r"\byani\b"
        }
        filler_counts = {}
        for key, regex in patterns.items():
            filler_counts[key] = len(re.findall(regex, transcript.lower()))

        # Düşük enerjili dolgu sesleri
        filler_audio_segments = detect_filler_like_segments(audio_path)
        filler_counts["ııı_eee"] = len(filler_audio_segments)
        filler_counts.pop("ııı", None)
        filler_counts.pop("eee", None)

        # Gaze analizi
        gaze_summary = analyze_gaze(fixed_video_path)

        # Ses tonu analizi
        tone_result = analyze_tone(audio_path)

        # Davranış analizi
        posture_result = analyze_posture(fixed_video_path)
        body_language_feedback = posture_result["posture_feedback"]

        # GPT içeriği
        prompt = f"""
Aşağıdaki sunum konuşmasını analiz et:
Kullanıcıya sen diliyle hitap et — üçüncü şahıs kullanma. Aşağıdaki iki başlık altında cevap ver:

{transcript}

Analiz formatı:
- Konunun özeti
- Sunumunun güçlü yönleri
- Geliştirilmesi gereken yönler
- Önerilen geliştirmeler
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        analysis = response.choices[0].message.content

        return {
            "message": "Video analiz tamamlandı.",
            "analysis": analysis,
            "filler_counts": filler_counts,
            "gaze": gaze_summary,
            "tone": tone_result,
            "body_language": body_language_feedback


        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Video yüklenemedi: {str(e)}"})

# Statik dosyaları yayınla
app.mount(f"/{CONVERTED_DIR}", StaticFiles(directory=CONVERTED_DIR), name="converted")
