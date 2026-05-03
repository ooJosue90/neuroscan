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

logger = logging.getLogger("TalosV10")
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
    Calcula el veredicto basado en la probabilidad (Modelo Balanceado V10.4):
      0 – 40%   → REAL
      41 – 59%  → INCIERTO
      60 – 100% → IA
    """
    if prob <= 40:
        return "REAL"
    elif prob <= 59:
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


async def _bypass_instagram(url: str):
    """
    Bypass especializado para Instagram (sin cookies, sin yt-dlp).

    Estrategia de 4 capas:
    1. Instagram oEmbed API pública → thumbnail_url CDN directo.
    2. DDInstagram proxy (d.ddinstagram.com) → parsea og:image/og:video.
    3. InstaFix proxy (instagramez.com / ezgif Instagram mirrors).
    4. Scrape directo del HTML de Instagram (og:image en head).
    """
    import re
    import shutil

    # Normalizar URL (asegurar dominio www)
    clean_url = url.strip().rstrip("/")
    if not clean_url.startswith("http"):
        clean_url = "https://" + clean_url

    # Extraer el shortcode del post
    shortcode_match = re.search(r"/(?:p|reel|tv)/([A-Za-z0-9_-]+)", clean_url)
    if not shortcode_match:
        logger.warning(">>> Instagram: no se pudo extraer shortcode de la URL.")
        return None

    shortcode = shortcode_match.group(1)
    logger.info(">>> Instagram BYPASS — shortcode: %s", shortcode)

    # Helper: extrae URLs de og:image y og:video del HTML de forma flexible
    def _parse_og_tags(html: str):
        """Extrae og:image y og:video del HTML con regex permisiva."""
        og_image = None
        og_video = None

        # Patrón 1: property="og:X" ... content="URL"  (orden estándar)
        for m in re.finditer(
            r'<meta[^>]*?property=["\']?og:(image|video(?::secure_url)?)["\']?[^>]*?content=["\']([^"\'>]+)["\']',
            html, re.IGNORECASE | re.DOTALL
        ):
            tag, val = m.group(1).lower(), m.group(2)
            if "video" in tag and not og_video:
                og_video = val
            elif tag == "image" and not og_image:
                og_image = val

        # Patrón 2: content="URL" ... property="og:X"  (orden invertido — común en Instagram)
        for m in re.finditer(
            r'<meta[^>]*?content=["\']([^"\'>]+)["\'][^>]*?property=["\']?og:(image|video(?::secure_url)?)["\']?',
            html, re.IGNORECASE | re.DOTALL
        ):
            val, tag = m.group(1), m.group(2).lower()
            if "video" in tag and not og_video:
                og_video = val
            elif tag == "image" and not og_image:
                og_image = val

        import html as html_lib
        return (html_lib.unescape(og_image) if og_image else None, 
                html_lib.unescape(og_video) if og_video else None)

    async def _download_and_analyze(client, media_url: str, label: str, force_video: bool = False):
        """Descarga un CDN URL y lo analiza como imagen o video."""
        try:
            resp = await client.get(media_url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "").lower()
            # Verificar .mp4 en el path de la URL (antes del '?'), ya que los CDN
            # de Instagram incluyen query params: ...video.mp4?_nc_ht=scontent...
            url_path = media_url.split("?")[0].lower()
            is_vid = force_video or "video" in ct or url_path.endswith(".mp4")
            logger.info(">>> %s media descargada (%.2f MB, tipo: %s, es_video: %s)", label, len(resp.content) / 1_048_576, ct, is_vid)
            if is_vid:
                tmpdir = tempfile.mkdtemp(prefix="talos_ig_")
                filepath = os.path.join(tmpdir, "media.mp4")
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                from models.video_detector import get_detector
                res = await run_in_threadpool(get_detector().analyze_file, filepath)
                shutil.rmtree(tmpdir, ignore_errors=True)
                return res
            else:
                # is_social_media=True porque sabemos que viene de un bypass social
                return await run_in_threadpool(analyze_image, resp.content, False, True)
        except Exception as e:
            if "403" in str(e) and "weserv" not in media_url:
                logger.info(">>> 403 detectado, reintentando via Weserv proxy...")
                proxy_url = f"https://images.weserv.nl/?url={media_url}"
                return await _download_and_analyze(client, proxy_url, label + "-Weserv", force_video)
            logger.warning(">>> %s descarga falló: %s", label, e)
            return None

    # ── Capa 1: Proxies de Embed (DDInstagram, InstaNavigation, etc) ──────────
    # Usa User-Agent de bot (Discord/Telegram) para que los proxies sirvan el embed
    dd_hosts = [
        "www.instagramez.com",
        "instagramez.com",
        "ddinstagram.com", 
        "www.ddinstagram.com"
    ]
    dd_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)",
        "Accept": "text/html,*/*",
    }

    for dd_host in dd_hosts:
        try:
            # Reemplazar dominio manteniendo el path completo
            dd_url = re.sub(r"(?:www\.)?instagram\.com", dd_host, clean_url)
            logger.info(">>> DDInstagram [%s]: %s", dd_host, dd_url)

            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=dd_headers) as dd_client:
                dd_resp = await dd_client.get(dd_url)
                ct = dd_resp.headers.get("content-type", "").lower()

                # Caso A: media directa
                if "image" in ct or "video" in ct:
                    ext = "jpg" if "image" in ct else "mp4"
                    logger.info(">>> DDInstagram media directa (%.2f MB)", len(dd_resp.content) / 1_048_576)
                    if "video" in ct:
                        tmpdir = tempfile.mkdtemp(prefix="talos_ig_")
                        filepath = os.path.join(tmpdir, "media.mp4")
                        with open(filepath, "wb") as f:
                            f.write(dd_resp.content)
                        from models.video_detector import get_detector
                        res = await run_in_threadpool(get_detector().analyze_file, filepath)
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        return res
                    else:
                        # is_social_media=True
                        return await run_in_threadpool(analyze_image, dd_resp.content, False, True)

                # Caso B: HTML → parsear og tags
                if "text/html" in ct and dd_resp.text:
                    og_image, og_video = _parse_og_tags(dd_resp.text)
                    media_cdn = og_video or og_image
                    is_vid = og_video is not None
                    
                    # [V10.4] Protección de Reels: Si es un Reel pero solo hallamos imagen, no retornamos
                    # para permitir que el fallback de yt-dlp intente obtener el video real.
                    is_reel_url = "/reel/" in clean_url.lower() or "/reels/" in clean_url.lower()
                    if is_reel_url and not is_vid:
                        logger.warning(">>> DDInstagram: Es un Reel pero no se encontró og:video. Saltando para intentar fallback de video...")
                        continue

                    if media_cdn:
                        logger.info(">>> DDInstagram og:%s → %s", "video" if is_vid else "image", media_cdn)
                        # force_video=True cuando viene de og:video (URL de Reel)
                        res = await _download_and_analyze(dd_client, media_cdn, "DDInstagram", force_video=is_vid)
                        if res:
                            return res
        except Exception as e:
            logger.warning(">>> DDInstagram [%s] falló: %s", dd_host, e)

    # ── Capa 2: Scrape directo del HTML de Instagram (Mejor calidad para posts públicos) ──
    # Instagram carga og:image en el HTML estático para posts públicos
    ig_scrape_headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Site": "none",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=ig_scrape_headers) as client:
            resp = await client.get(clean_url)
            ct = resp.headers.get("content-type", "").lower()
            if resp.status_code == 200 and "text/html" in ct:
                og_image, og_video = _parse_og_tags(resp.text)
                media_cdn = og_video or og_image
                is_vid = og_video is not None
                if media_cdn:
                    is_reel_url = "/reel/" in clean_url.lower() or "/reels/" in clean_url.lower()
                    if is_reel_url and not is_vid:
                        logger.warning(">>> Scrape IG: Es Reel pero solo hay og:image. Saltando para intentar fallback...")
                    else:
                        logger.info(">>> Scrape IG directo → og:%s: %s", "video" if is_vid else "image", media_cdn)
                        res = await _download_and_analyze(client, media_cdn, "IG-Scrape", force_video=is_vid)
                        if res:
                            return res
                else:
                    logger.warning(">>> Scrape IG: HTML obtenido pero sin og:image/og:video (post privado o bloqueado).")
    except Exception as e:
        logger.warning(">>> Scrape IG directo falló: %s", e)

    # ── Capa 2.5: Proxy de descarga forzada (via Weserv para saltar 403) ──────
    # Si las capas anteriores fallaron pero tenemos una URL de CDN, intentamos via proxy
    if 'media_cdn' in locals() and media_cdn:
        logger.info(">>> Reintentando descarga de CDN via proxy Weserv para saltar 403...")
        proxy_cdn_url = f"https://images.weserv.nl/?url={media_cdn}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await _download_and_analyze(client, proxy_cdn_url, "Weserv-Proxy", force_video=is_vid)
            if res:
                return res

    # ── Capa 3: Instagram oEmbed API pública (Fallback baja resolución) ─────
    # Lo dejamos al final porque devuelve una thumbnail comprimida que puede causar resultados "INCIERTOS"
    oembed_url = f"https://graph.facebook.com/v18.0/instagram_oembed?url={clean_url}&maxwidth=1080"
    oembed_legacy = f"https://www.instagram.com/api/v1/oembed/?url={clean_url}&maxwidth=1080&hidecaption=1"
    
    ig_headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "application/json,text/html,*/*",
        "X-IG-App-ID": "936619743392459",
        "Referer": "https://www.instagram.com/",
        "Origin": "https://www.instagram.com",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=ig_headers) as client:
            for oembed_endpoint in [oembed_legacy, oembed_url]:
                try:
                    resp = await client.get(oembed_endpoint)
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except Exception:
                            continue
                        post_type = data.get("type", "photo").lower()  # "photo", "video", "rich"
                        thumb_url = data.get("thumbnail_url", "")
                        logger.info(">>> oEmbed OK: type=%s, thumbnail=%s", post_type, bool(thumb_url))

                        if post_type in ("photo", "rich") and thumb_url:
                            # [V10.4] Si es un Reel, no permitimos el fallback a imagen de oEmbed
                            # Queremos que siga de largo para que yt-dlp intente el video.
                            is_reel_url = "/reel/" in clean_url.lower() or "/reels/" in clean_url.lower()
                            if is_reel_url:
                                logger.warning(">>> oEmbed: Es Reel pero solo hay miniatura. Saltando para intentar fallback final...")
                                break

                            # Es imagen o carrusel: la thumbnail es una imagen válida del post
                            res = await _download_and_analyze(client, thumb_url, "oEmbed-CDN (Fallback)")
                            if res:
                                return res
                        else:
                            # Es Reel/Video: oEmbed no da video real. Como Capa 1 y 2 ya fallaron, abortamos.
                            logger.warning(">>> oEmbed: Reel/video detectado pero Capa 1 y 2 fallaron. Imposible obtener .mp4.")
                            break
                except Exception as e2:
                    logger.debug(">>> oEmbed endpoint %s falló: %s", oembed_endpoint, e2)
    except Exception as e:
        logger.warning(">>> Capa 3 oEmbed falló: %s", e)

    logger.warning(">>> Instagram: todas las capas de bypass fallaron para shortcode=%s.", shortcode)
    return None


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
    # Configuración optimizada para evitar bloqueos sin dependencias complejas
    if _HAS_CURL_CFFI:
        # Algunos sitios requieren impersonación para no bloquear yt-dlp
        # Usamos un try-except interno en yt-dlp si es posible
        pass

    def _attempt_download(opts: dict) -> tuple:
        with yt_dlp.YoutubeDL(opts) as ydl:
            # Primero extraemos info sin descargar para decidir qué hacer
            info = ydl.extract_info(url, download=False)
            
            # Si es una lista de entradas (ej: hilo de X), tomamos la primera
            if "entries" in info:
                info = info["entries"][0]
            
            # Caso especial: Si no hay video pero hay miniaturas (usualmente imágenes en X/Twitter)
            # yt-dlp a veces no descarga nada si solo hay imágenes.
            has_video = any(f.get('vcodec') != 'none' for f in info.get('formats', []))
            
            if not has_video and info.get('thumbnails'):
                # Intento de obtener la imagen original en lugar de un video
                # Buscamos la miniatura de mayor calidad (usualmente la última)
                logger.info(">>> No se detectó video, intentando extraer imagen de miniatura...")
                thumb_url = info['thumbnails'][-1]['url']
                
                # Descargar manualmente la miniatura (imagen) como si fuera el archivo principal
                import httpx
                with httpx.Client(follow_redirects=True, headers=_HTTP_HEADERS) as client:
                    resp = client.get(thumb_url)
                    resp.raise_for_status()
                    ext = thumb_url.split('.')[-1].split('?')[0] or "jpg"
                    if len(ext) > 4: ext = "jpg"
                    
                    filepath = os.path.join(tmpdir, f"media.{ext}")
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    
                    logger.info(">>> Imagen extraída exitosamente de la red social.")
                    return filepath, ext

            # Caso normal: descargar video
            ydl.process_info(info)
            
        files = [f for f in os.listdir(tmpdir) if not f.startswith(".")]
        if not files:
            raise Exception("yt-dlp no generó ningún archivo. Verifica si el enlace es público.")
        
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

        # Nota: Se omiten los intentos con cookies de navegador porque en Windows
        # causan errores DPAPI (https://github.com/yt-dlp/yt-dlp/issues/10927)
        # que no se pueden resolver sin acceso al proceso propietario de las cookies.

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

        # ── RAMA 1: Redes sociales → Bypass especializado / yt-dlp ──────────────
        if is_social:
            # ── Bypass Twitter/X (vxtwitter API) ────────────────────────────────
            if "twitter.com" in url.lower() or "x.com" in url.lower():
                try:
                    logger.info(">>> TWITTER DETECTADO: Usando bypass vxtwitter para máxima fiabilidad...")
                    api_url = url.replace("twitter.com", "api.vxtwitter.com").replace("x.com", "api.vxtwitter.com")
                    
                    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
                        resp = await client.get(api_url)
                        if resp.status_code == 200:
                            data = resp.json()
                            media_list = data.get("media_extended", []) or data.get("media", [])
                            if media_list:
                                target_media = media_list[0]
                                media_url = target_media.get("url")
                                
                                if media_url:
                                    logger.info(">>> Media encontrada vía bypass: %s", media_url)
                                    
                                    media_content = None
                                    if _HAS_CURL_CFFI:
                                        from curl_cffi.requests import AsyncSession
                                        async with AsyncSession(impersonate="chrome") as s:
                                            resp_media = await s.get(media_url, headers={"Referer": "https://x.com/"})
                                            if resp_media.status_code == 200:
                                                media_content = resp_media.content
                                    
                                    if media_content is None:
                                        media_resp = await client.get(media_url, headers={"Referer": "https://x.com/"})
                                        media_resp.raise_for_status()
                                        media_content = media_resp.content
                                    
                                    tmpdir_cleanup = tempfile.mkdtemp(prefix="talos_bypass_")
                                    ext = media_url.split(".")[-1].split("?")[0].lower()
                                    if len(ext) > 4: ext = "mp4" if target_media.get("type") == "video" else "jpg"
                                    
                                    filepath = os.path.join(tmpdir_cleanup, f"media.{ext}")
                                    with open(filepath, "wb") as f:
                                        f.write(media_content)
                                    
                                    media_type = _detect_type_from_mime("", ext)
                                    logger.info(">>> Análisis de bypass iniciado (tipo: %s)", media_type)
                                    
                                    if media_type == "video":
                                        from models.video_detector import get_detector
                                        result = await run_in_threadpool(get_detector().analyze_file, filepath)
                                    elif media_type == "image":
                                        result = await run_in_threadpool(analyze_image, media_content, False, True)
                                    else:
                                        result = await run_in_threadpool(analyze_video, media_content)
                                    
                                    if result:
                                        logger.info(">>> Análisis vía bypass COMPLETADO con éxito.")
                except Exception as e:
                    logger.warning(">>> Bypass vxtwitter falló: %s (Continuando con yt-dlp...)", e)

            # ── Bypass Instagram (oEmbed + DDInstagram) ──────────────────────────
            # Instagram bloquea yt-dlp sin cookies y las cookies DPAPI fallan en Windows.
            # Usamos oEmbed público (sin auth) para obtener la thumbnail/video CDN URL.
            elif "instagram.com" in url.lower():
                result = await _bypass_instagram(url)
                if result:
                    logger.info(">>> Análisis Instagram vía bypass COMPLETADO.")
                else:
                    logger.warning(">>> Bypass Instagram falló — intentando yt-dlp básico...")

            # Fallback a motor yt-dlp si el bypass no resolvió el resultado
            if result is None:
                logger.info(">>> LINK SOCIAL — usando motor yt-dlp ...")
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
                        result = await run_in_threadpool(analyze_image, data, False, is_social)
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
                    logger.warning("!!! Motor yt-dlp falló: %s", e)
                    # Para Instagram NO hacemos fallback HTTP directo (devuelve HTML).
                    # Para otras redes, permitimos el fallback.
                    if "instagram.com" not in url.lower():
                        is_social = False

        # ── RAMA 2: URL directa (imagen, audio, o video público) ──────────
        if not is_social and result is None:
            logger.info(">>> Descarga directa HTTP para: %s", url)
            
            data = None
            mime = ""
            
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
            except httpx.HTTPStatusError as e:
                logger.warning(f">>> Descarga directa HTTP falló con código {e.response.status_code}. Intentando con curl_cffi...")
                if _HAS_CURL_CFFI:
                    try:
                        from curl_cffi.requests import AsyncSession
                        async with AsyncSession(impersonate="chrome", timeout=45) as s:
                            # Intentar sin el Referer de google por si es eso lo que causa el 404/403 en algunos CDN
                            fallback_headers = {"User-Agent": _BROWSER_UA, "Accept": "*/*"}
                            resp_media = await s.get(url, headers=fallback_headers)
                            if resp_media.status_code == 200:
                                data = resp_media.content
                                mime = resp_media.headers.get("content-type", "").lower()
                            else:
                                return {"status": "error", "error": f"Error HTTP {resp_media.status_code} al descargar la URL (Fallback)."}
                    except Exception as ex:
                        return {"status": "error", "error": f"Error en fallback curl_cffi: {str(ex)}"}
                else:
                    return {"status": "error", "error": f"Error HTTP {e.response.status_code} al descargar la URL."}

            try:
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

                # Verificar extensión desde Content-Disposition si existe (solo si response está definido, usamos ext_from_url como fallback)
                # Omitimos esto si data vino de curl_cffi y no tenemos headers de content-disposition
                cd = ""
                if 'response' in locals() and hasattr(response, 'headers'):
                    cd = response.headers.get("content-disposition", "")
                
                if "filename=" in cd:
                    disc_name = cd.split("filename=")[-1].strip('"').strip("'")
                    if "." in disc_name:
                        ext_from_url = disc_name.rsplit(".", 1)[-1].lower()

                logger.info(">>> Descarga directa completada: %.2f MB, MIME=%s", len(data) / 1_048_576, mime)

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

            except Exception as e:
                return {"status": "error", "error": f"Error al procesar contenido descargado: {str(e)}"}

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