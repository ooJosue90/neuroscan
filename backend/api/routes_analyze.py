from fastapi import APIRouter, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from models.image_detector import analyze_image
from models.video_detector import analyze_video
from models.audio_detector import analyze_audio

router = APIRouter()

@router.post("/analizar")
async def analyze(file: UploadFile = File(...)):

    data = await file.read()
    mime = file.content_type

    if mime.startswith("image/"):
        return await run_in_threadpool(analyze_image, data)

    elif mime.startswith("video/"):
        return await run_in_threadpool(analyze_video, data)

    elif mime.startswith("audio/"):
        return await run_in_threadpool(analyze_audio, data)

    return {"error": "tipo no soportado"}