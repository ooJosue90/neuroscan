from fastapi import APIRouter, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from models.image_detector import analyze_image
from models.video_detector import analyze_video
from models.audio_detector import analyze_audio
from fastapi import Request
import httpx
import tempfile
import os
import mimetypes
import yt_dlp
import traceback
import logging

logger = logging.getLogger("NueroscanV10")
router = APIRouter()

# ── User-Agent de navegador real para evitar bloqueos ─────────────────────
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_HTTP_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "image/*,video/*,audio/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# ── Redes sociales que requieren yt-dlp ───────────────────────────────────
_SOCIAL_DOMAINS = [
    "youtube", "youtu.be", "tiktok", "instagram", "twitter",
    "x.com", "facebook", "reddit", "vimeo", "twitch", "bilibili",
    "dailymotion", "rumble", "odysee", "snapchat", "pinterest"
]

# ── Extensiones soportadas por tipo ───────────────────────────────────────
_VIDEO_EXTS = {"mp4", "avi", "mov", "webm", "mkv", "flv", "m4v", "3gp", "ts"}
_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "avif"}
_AUDIO_EXTS = {"wav", "mp3", "ogg", "m4a", "flac", "aac", "opus", "wma"}

# ── Detectar si curl-cffi está disponible para impersonación TLS ──────────
try:
    import curl_cffi  # noqa: F401
    _HAS_CURL_CFFI = True
    logger.info("curl-cffi disponible — impersonación Chrome habilitada para yt-dlp")
except ImportError:
    _HAS_CURL_CFFI = False
    logger.warning("curl-cffi no encontrado — instala con: pip install curl-cffi")


def _compute_verdict(prob: float) -> str:
    """
    Modelo Balanceado TALOS:
      0  – 40%  → REAL
      41 – 69%  → INCIERTO
      70 – 100% → IA
    """
    if prob <= 40:
        return "REAL"
    elif prob <= 69:
        return "INCIERTO"
    return "IA"


def _detect_type_from_mime(mime: str, ext: str):
    """Retorna 'video', 'image', 'audio' o None según MIME y extensión."""
    mime = mime.lower().split(";")[0].strip()
    if mime.startswith("video/") or ext in _VIDEO_EXTS:
        return "video"
    if mime.startswith("image/") or ext in _IMAGE_EXTS:
        return "image"
    if mime.startswith("audio/") or ext in _AUDIO_EXTS:
        return "audio"
    return None


def _patch_verdict(result: dict) -> dict:
    """Recalcula el veredicto aplicando el modelo Balanceado al resultado final."""
    prob = result.get("probabilidad", 0)
    if prob is not None:
        result["verdict"] = _compute_verdict(float(prob))
    return result


def download_with_ytdlp(url: str):
    """
    Descarga contenido multimedia usando yt-dlp.

    MEJORAS V10.4:
    - Impersonación Chrome via curl-cffi (elimina WARNING de TikTok/IG).
    - Primer intento con impersonación (sin cookies), más rápido y sigiloso.
    - Fallback a cookies del navegador si la impersonación también falla.
    - Descarga a 720p máx para velocidad óptima.
    - Retorna (ruta_archivo, extensión, tmpdir) — Zero-Copy compatible.
    """
    tmpdir = tempfile.mkdtemp(prefix="talos_ytdlp_")

    base_opts = {
        "outtmpl": os.path.join(tmpdir, "media.%(ext)s"),
        # 720p es suficiente para detectar artefactos de deepfake.
        # Reduce el tamaño de descarga ~3-5x en contenido 1080p/4K.
        "format": (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=720][ext=mp4]/"
            "best[height<=720]/best"
        ),
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "ignoreerrors": False,
        "logtostderr": False,
        "max_filesize": 100 * 1024 * 1024,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
        "http_headers": {
            "User-Agent": _BROWSER_UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        },
    }

    # Activar impersonación Chrome si curl-cffi está disponible.
    # Esto hace pasar yt-dlp como Chrome real a nivel de TLS (JA3/ALPN),
    # eliminando el WARNING y el bloqueo de TikTok/Instagram en el primer intento.
    if _HAS_CURL_CFFI:
        base_opts["impersonate"] = "chrome"

    def _attempt_download(opts: dict) -> tuple:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
        files = [f for f in os.listdir(tmpdir) if not f.startswith(".")]
        if not files:
            raise Exception("yt-dlp no generó ningún archivo.")
        filepath = os.path.join(tmpdir, files[0])
        ext = files[0].rsplit(".", 1)[-1].lower() if "." in files[0] else "mp4"
        logger.info(">>> yt-dlp descargó: %s (%.1f MB)", files[0],
                    os.path.getsize(filepath) / 1_048_576)
        return filepath, ext

    try:
        # Intento 1: Configuración base (incluye impersonación si está disponible)
        try:
            return (*_attempt_download(base_opts), tmpdir)
        except Exception as e1:
            logger.warning("yt-dlp Intento 1 falló: %s — reintentando sin impersonación...", e1)
            # Limpiar descargas parciales
            for f in os.listdir(tmpdir):
                try:
                    os.remove(os.path.join(tmpdir, f))
                except Exception:
                    pass

            # Crear configuración sin impersonate para los fallbacks
            base_opts_fallback = dict(base_opts)
            base_opts_fallback.pop("impersonate", None)

            # Intento 1.5: Sin impersonación, sin cookies (por si el impersonate fue lo que falló)
            if "impersonate" in base_opts:
                try:
                    logger.info("Intentando extracción sin impersonación...")
                    return (*_attempt_download(base_opts_fallback), tmpdir)
                except Exception as e1_5:
                    logger.warning("yt-dlp sin impersonación falló: %s", e1_5)

        # Intento 2: Con cookies del navegador usando la config sin impersonación
        # Si impersonate falló, mezclarlo con cookies también fallará.
        if "base_opts_fallback" not in locals():
            base_opts_fallback = dict(base_opts)
            base_opts_fallback.pop("impersonate", None)

        browsers = ["firefox", "brave", "edge", "chrome"]
        for browser in browsers:
            try:
                logger.info("Intentando extracción con cookies de %s...", browser)
                opts_cookies = {**base_opts_fallback, "cookiesfrombrowser": (browser,)}
                return (*_attempt_download(opts_cookies), tmpdir)
            except Exception as e_cookie:
                logger.debug("Fallo con cookies de %s: %s", browser, e_cookie)
                continue

    except Exception:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    raise Exception("yt-dlp: todos los intentos de descarga fallaron.")


# ═══════════════════════════════════════════════════════════════
# RUTA: /analizar (subida de archivo local)
# ═══════════════════════════════════════════════════════════════
@router.post("/analizar")
async def analyze(file: UploadFile = File(...)):
    print(f">>> RECIBIENDO ARCHIVO: {file.filename} (MIME: {file.content_type})")
    try:
        data = await file.read()
        mime = file.content_type or ""
        ext  = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""

        media_type = _detect_type_from_mime(mime, ext)

        if media_type == "image":
            result = await run_in_threadpool(analyze_image, data)
        elif media_type == "video":
            result = await run_in_threadpool(analyze_video, data)
        elif media_type == "audio":
            result = await run_in_threadpool(analyze_audio, data)
        else:
            return {"status": "error", "error": f"Formato '{mime or ext}' no soportado por el clúster."}

        return _patch_verdict(result)

    except Exception as e:
        logger.error("!!! ERROR EN API /analizar: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# RUTA: /analizar-url (análisis desde enlace público)
# ═══════════════════════════════════════════════════════════════
@router.post("/analizar-url")
async def analyze_url(request: Request):
    """
    Pipeline de análisis por URL V10.4:

    1. Link de red social → yt-dlp con impersonación Chrome (Zero-Copy).
    2. Link directo de imagen/audio → descarga httpx → pipeline respectivo.
    3. Veredicto recalibrado con modelo Balanceado (INCIERTO 41-69%).
    """
    import shutil
    tmpdir_cleanup = None

    try:
        body = await request.json()
        url  = (body.get("url") or "").strip()
        if not url:
            return {"status": "error", "error": "URL no proporcionada."}

        logger.info(">>> ANALIZANDO URL: %s", url)

        # Detectar extensión base de la URL (antes del '?')
        url_clean = url.split("?")[0].rstrip("/")
        ext_from_url = url_clean.rsplit(".", 1)[-1].lower() if "." in url_clean else ""

        is_social = any(d in url.lower() for d in _SOCIAL_DOMAINS)

        result = None

        # ── RAMA 1: Redes sociales → yt-dlp ──────────────────────────────
        if is_social:
            logger.info(">>> LINK SOCIAL detectado — usando motor yt-dlp ...")
            try:
                filepath, ext, tmpdir_cleanup = await run_in_threadpool(
                    download_with_ytdlp, url
                )

                media_type = _detect_type_from_mime("", ext)
                logger.info(">>> Tipo detectado por yt-dlp: %s (ext=%s)", media_type, ext)

                if media_type == "video":
                    # Zero-Copy: pasar ruta al pipeline directamente
                    from models.video_detector import get_detector
                    result = await run_in_threadpool(
                        get_detector().analyze_file, filepath
                    )
                elif media_type == "image":
                    with open(filepath, "rb") as f:
                        data = f.read()
                    result = await run_in_threadpool(analyze_image, data)
                elif media_type == "audio":
                    with open(filepath, "rb") as f:
                        data = f.read()
                    result = await run_in_threadpool(analyze_audio, data)
                else:
                    # Fallback: intentar como video
                    from models.video_detector import get_detector
                    result = await run_in_threadpool(
                        get_detector().analyze_file, filepath
                    )

                logger.info(">>> Análisis yt-dlp completado.")

            except Exception as e:
                logger.warning("!!! Motor yt-dlp falló: %s — intentando descarga directa.", e)
                is_social = False  # Forzar fallback a HTTP directo

        # ── RAMA 2: URL directa (imagen, audio, o video público) ──────────
        if not is_social and result is None:
            logger.info(">>> Descarga directa HTTP para: %s", url)
            try:
                async with httpx.AsyncClient(
                    timeout=45.0,
                    follow_redirects=True,
                    headers=_HTTP_HEADERS,
                ) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.content
                    mime = response.headers.get("content-type", "").lower()

                # Guard: Si la respuesta es HTML, no es un archivo multimedia
                if "text/html" in mime or "text/plain" in mime:
                    return {
                        "status": "error",
                        "error": (
                            "La URL devuelve una página web, no un archivo multimedia. "
                            "Para redes sociales (TikTok, Instagram), asegúrate de que "
                            "el enlace sea directo y público."
                        )
                    }

                # Verificar extensión desde Content-Disposition si existe
                cd = response.headers.get("content-disposition", "")
                if "filename=" in cd:
                    disc_name = cd.split("filename=")[-1].strip('"').strip("'")
                    if "." in disc_name:
                        ext_from_url = disc_name.rsplit(".", 1)[-1].lower()

                logger.info(">>> Descarga directa: %.2f MB, MIME=%s", len(data) / 1_048_576, mime)

                media_type = _detect_type_from_mime(mime, ext_from_url)

                if media_type == "image":
                    result = await run_in_threadpool(analyze_image, data)
                elif media_type == "video":
                    result = await run_in_threadpool(analyze_video, data)
                elif media_type == "audio":
                    result = await run_in_threadpool(analyze_audio, data)
                else:
                    # Último recurso: intentar como imagen
                    try:
                        result = await run_in_threadpool(analyze_image, data)
                    except Exception:
                        return {
                            "status": "error",
                            "error": (
                                f"No se pudo determinar el tipo de medio: '{mime or ext_from_url}'. "
                                "Asegúrate de que la URL apunte directamente a un archivo multimedia."
                            )
                        }

            except httpx.HTTPStatusError as e:
                return {"status": "error", "error": f"Error HTTP {e.response.status_code} al descargar la URL."}
            except Exception as e:
                return {"status": "error", "error": f"Error al descargar contenido: {str(e)}"}

        if result is None:
            return {"status": "error", "error": "No se pudo analizar el contenido desde la URL proporcionada."}

        return _patch_verdict(result)

    except Exception as e:
        logger.error("!!! ERROR GLOBAL EN /analizar-url: %s", e, exc_info=True)
        return {"status": "error", "error": f"Error interno del clúster: {str(e)}"}

    finally:
        # Limpiar directorio temporal de yt-dlp
        if tmpdir_cleanup and os.path.isdir(tmpdir_cleanup):
            try:
                import shutil
                shutil.rmtree(tmpdir_cleanup, ignore_errors=True)
                logger.info(">>> Temporal yt-dlp limpiado: %s", tmpdir_cleanup)
            except Exception:
                pass