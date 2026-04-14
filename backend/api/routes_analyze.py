from fastapi import APIRouter, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from models.image_detector import analyze_image
from models.video_detector import analyze_video
from models.audio_detector import analyze_audio

router = APIRouter()

@router.post("/analizar")
async def analyze(file: UploadFile = File(...)):
    print(f">>> RECIBIENDO ARCHIVO: {file.filename} (MIME: {file.content_type})")
    try:
        data = await file.read()
        mime = file.content_type or ""
        ext = file.filename.split('.')[-1].lower() if file.filename else ""

        if mime.startswith("image/") or ext in ['jpg', 'jpeg', 'png', 'webp']:
            return await run_in_threadpool(analyze_image, data)

        elif mime.startswith("video/") or ext in ['mp4', 'avi', 'mov', 'webm']:
            return await run_in_threadpool(analyze_video, data)

        elif mime.startswith("audio/") or ext in ['wav', 'mp3', 'ogg', 'm4a', 'flac']:
            return await run_in_threadpool(analyze_audio, data)

        return {"status": "error", "error": f"Formato '{mime or ext}' no soportado por el clúster."}
    except Exception as e:
        print(f"!!! ERROR EN API: {e}")
        return {"status": "error", "error": str(e)}