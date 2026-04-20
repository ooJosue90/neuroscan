if (!localStorage.getItem('talos_session')) {
    window.location.href = 'login.html';
}

window.logout = function () {
    localStorage.removeItem('talos_session');
    window.location.href = 'login.html';
};

let wavesurfer;

const fileInput = document.getElementById('fileInput'),
    dropZone = document.getElementById('dropZone'),
    terminal = document.getElementById('terminal'),
    scanBtn = document.getElementById('scanBtn'),
    btnHelp = document.getElementById('btnHelp'),
    closeHelp = document.getElementById('closeHelp'),
    helpModal = document.getElementById('helpModal');

const toggleModal = (show) => {
    if (show) {
        helpModal.classList.add('active');
        if (closeHelp) closeHelp.focus();

        // Ocultar todos los manuals primero
        const helpDivs = ['help-scanner', 'help-database', 'help-contact', 'help-academy', 'help-challenge'];
        helpDivs.forEach(id => {
            const div = document.getElementById(id);
            if (div) div.classList.add('d-none');
        });

        // Detectar sección activa
        if (document.getElementById('nav-scanner').classList.contains('active')) {
            const isUrlMode = !document.getElementById('ui-url-scan').classList.contains('d-none');
            const isFileMode = !document.getElementById('ui-file-scan').classList.contains('d-none');

            const helpDivs = ['help-scanner', 'help-file-scan', 'help-url-scan', 'help-database', 'help-contact', 'help-academy', 'help-challenge'];
            helpDivs.forEach(id => {
                const div = document.getElementById(id);
                if (div) div.classList.add('d-none');
            });

            if (isUrlMode) {
                document.getElementById('helpTitle').innerText = "MANUAL: EXTRACCIÓN URL";
                document.getElementById('help-url-scan').classList.remove('d-none');
            } else if (isFileMode) {
                document.getElementById('helpTitle').innerText = "MANUAL: ESCÁNER LOCAL";
                document.getElementById('help-file-scan').classList.remove('d-none');
            } else {
                document.getElementById('helpTitle').innerText = "MANUAL DEL ESCÁNER";
                document.getElementById('help-scanner').classList.remove('d-none');
            }
        } else if (document.getElementById('nav-database').classList.contains('active')) {
            document.getElementById('helpTitle').innerText = "SISTEMA DE REGISTROS";
            document.getElementById('help-database').classList.remove('d-none');
        } else if (document.getElementById('nav-contact').classList.contains('active')) {
            document.getElementById('helpTitle').innerText = "PROTOCOLOS DE CONTACTO";
            document.getElementById('help-contact').classList.remove('d-none');
        } else if (document.getElementById('nav-academy').classList.contains('active')) {
            document.getElementById('helpTitle').innerText = "KNOWLEDGE HUB ACADEMY";
            document.getElementById('help-academy').classList.remove('d-none');
        } else if (document.getElementById('nav-challenge').classList.contains('active')) {
            document.getElementById('helpTitle').innerText = "CALIFICACIÓN FORENSE";
            document.getElementById('help-challenge').classList.remove('d-none');
        }
    } else {
        helpModal.classList.remove('active');
        btnHelp.focus();
    }
};

btnHelp.onclick = () => toggleModal(true);
closeHelp.onclick = () => toggleModal(false);
const closeHelpBtn = document.getElementById('closeHelpBtn');
if (closeHelpBtn) closeHelpBtn.onclick = () => toggleModal(false);

helpModal.onclick = (e) => {
    if (e.target === helpModal) toggleModal(false);
};
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && helpModal.classList.contains('active')) toggleModal(false);
});

const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarBackdrop = document.getElementById('sidebarBackdrop');

if (sidebarToggle) {
    sidebarToggle.onclick = (e) => {
        e.stopPropagation();
        const isOpen = sidebar.classList.toggle('open');
        sidebar.classList.toggle('collapsed');
        document.body.classList.toggle('sidebar-collapsed');

        if (sidebarBackdrop) {
            if (isOpen && window.innerWidth <= 991) {
                sidebarBackdrop.classList.add('active');
            } else {
                sidebarBackdrop.classList.remove('active');
            }
        }
    };
}

if (sidebarBackdrop) {
    sidebarBackdrop.onclick = () => {
        sidebar.classList.remove('open');
        sidebar.classList.add('collapsed');
        document.body.classList.add('sidebar-collapsed');
        sidebarBackdrop.classList.remove('active');
    };
}

// Cerrar sidebar al hacer clic fuera en móviles
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 992 &&
        !sidebar.contains(e.target) &&
        !sidebarToggle.contains(e.target) &&
        sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
    }
});

/* ----- Lógica Principal ----- */
function addLog(t) {
    const term = document.getElementById('terminal');
    if (!term) return;
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    term.innerHTML += `<br>> [${time}] ${t}`;
    term.scrollTop = term.scrollHeight;
}

function clearLogs() {
    console.clear();
    const term = document.getElementById('terminal');
    if (term) {
        // Borrado atómico
        while (term.firstChild) {
            term.removeChild(term.firstChild);
        }
        term.innerHTML = '> SISTEMA NEURO-FORENSE LISTO...<br>> SELECCIONE MODO DE ESCANEO...';
        term.scrollTop = 0;
    }
}

window.selectScanMode = function (mode) {
    clearLogs();
    document.getElementById('scan-mode-selection').classList.add('d-none');
    if (mode === 'file') {
        document.getElementById('ui-file-scan').classList.remove('d-none');
        document.getElementById('ui-url-scan').classList.add('d-none');
        addLog("MODO ACTIVADO: ARCHIVO LOCAL. EN ESPERA DE EVIDENCIA...");
    } else {
        document.getElementById('ui-url-scan').classList.remove('d-none');
        document.getElementById('ui-file-scan').classList.add('d-none');
        addLog("MODO ACTIVADO: EXTRACCIÓN POR URL. INSERTE EL ENLACE...");
    }
};

window.resetScanMode = function () {
    const modeSelection = document.getElementById('scan-mode-selection');
    if (modeSelection) modeSelection.classList.remove('d-none');

    const uiFileScan = document.getElementById('ui-file-scan');
    if (uiFileScan) uiFileScan.classList.add('d-none');

    const uiUrlScan = document.getElementById('ui-url-scan');
    if (uiUrlScan) uiUrlScan.classList.add('d-none');

    const urlInput = document.getElementById('urlInput');
    if (urlInput) urlInput.value = '';
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';

    // Limpiar URL preview card
    const urlPreviewCard = document.getElementById('urlPreviewCard');
    if (urlPreviewCard) urlPreviewCard.classList.add('d-none');
    const urlPreviewContent = document.getElementById('urlPreviewContent');
    if (urlPreviewContent) urlPreviewContent.classList.add('d-none');
    const urlPreviewLoading = document.getElementById('urlPreviewLoading');
    if (urlPreviewLoading) urlPreviewLoading.classList.remove('d-none');

    // Restaurar header del modo URL
    const urlScanHeader = document.getElementById('urlScanHeader');
    if (urlScanHeader) {
        urlScanHeader.style.maxHeight = '300px';
        urlScanHeader.style.opacity = '1';
        urlScanHeader.style.marginBottom = '';
        urlScanHeader.style.overflow = '';
        urlScanHeader.style.pointerEvents = '';
    }

    const uploadContent = document.getElementById('uploadContent');
    if (uploadContent) {
        uploadContent.style.setProperty('display', 'flex', 'important');
        uploadContent.classList.remove('d-none');
    }

    const imgPreview = document.getElementById('imgPreview');
    if (imgPreview) { imgPreview.classList.add('d-none'); imgPreview.src = ''; }

    const videoPreview = document.getElementById('videoPreview');
    if (videoPreview) { videoPreview.classList.add('d-none'); videoPreview.src = ''; }

    const audioWrapper = document.getElementById('audioWrapper');
    if (audioWrapper) audioWrapper.style.display = 'none';

    if (typeof wavesurfer !== 'undefined' && wavesurfer) { try { wavesurfer.destroy(); wavesurfer = null; } catch (e) { } }

    clearLogs();
};

// Soporte de Teclado Accesible para la zona de carga
dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(n => {
    dropZone.addEventListener(n, e => { e.preventDefault(); e.stopPropagation(); });
});
dropZone.addEventListener('dragover', () => dropZone.classList.add('drag-over'));
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
    dropZone.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
});

dropZone.onclick = (e) => {
    if (e.target.tagName !== 'VIDEO' && !e.target.closest('#audioWrapper')) fileInput.click();
};
fileInput.onchange = () => handleFile(fileInput.files[0]);

function handleFile(file) {
    if (!file) return;
    clearLogs();

    // Validación de archivos multimedia
    const allowedTypes = ['image/', 'video/', 'audio/'];
    const isAllowed = allowedTypes.some(type => file.type.startsWith(type));

    if (!isAllowed) {
        addLog(`!!! RECHAZADO: FORMATO NO SOPORTADO (${file.name.toUpperCase()})`);
        alert(`Talos rechazó la muestra: "${file.name}" no es un formato multimedia válido (Imagen, Video o Audio).`);
        fileInput.value = '';
        return;
    }

    fileInput.files = new DataTransfer().files; // Reset inner files if any
    const dt = new DataTransfer(); dt.items.add(file); fileInput.files = dt.files;

    const uploadContent = document.getElementById('uploadContent');
    const imgPreview = document.getElementById('imgPreview');
    const videoPreview = document.getElementById('videoPreview');
    const audioWrapper = document.getElementById('audioWrapper');

    uploadContent.style.setProperty('display', 'none', 'important');
    const url = URL.createObjectURL(file);

    imgPreview.classList.add('d-none');
    videoPreview.classList.add('d-none');
    audioWrapper.style.display = 'none';
    if (wavesurfer) wavesurfer.destroy();

    addLog(`EVIDENCIA CARGADA: ${file.name.toUpperCase()}`);

    if (file.type.startsWith('image/')) {
        const img = document.getElementById('imgPreview'); img.src = url; img.classList.remove('d-none');
    } else if (file.type.startsWith('video/')) {
        const vid = document.getElementById('videoPreview'); vid.src = url; vid.classList.remove('d-none');
    } else if (file.type.startsWith('audio/')) {
        const wrapper = document.getElementById('audioWrapper');
        wrapper.style.display = 'block';

        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'rgba(0, 255, 204, 0.2)',
            progressColor: '#00ffcc',
            cursorColor: '#ff0055',
            cursorWidth: 2,
            barWidth: 3,
            barGap: 2,
            barRadius: 2,
            height: 60
        });
        wavesurfer.load(url);
        document.getElementById('playBtn').onclick = () => wavesurfer.playPause();
    }
}

async function analyzeFile() {
    if (scanBtn.disabled) return;
    clearLogs();
    const file = fileInput.files[0];

    scanBtn.disabled = true;
    scanBtn.innerText = "ANALIZANDO EVIDENCIA...";
    document.getElementById('scanner').style.display = 'block';
    addLog(`INICIANDO ANÁLISIS FORENSE: ${file.type.toUpperCase()}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://127.0.0.1:8000/analizar', { method: 'POST', body: formData });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: Error en el servidor.`);
        }

        const data = await response.json();

        if (data.status === "error") {
            addLog(`!!! ERROR: ${data.error}`);
            alert(`Talos rechazó la muestra: ${data.error}`);
            return;
        }

        const finalVerdict = data.verdict || (data.probabilidad >= 50 ? 'IA' : 'REAL');
        renderResult(data);
        saveToHistory(file.name, finalVerdict, data);
    } catch (err) {
        console.error(err);
        addLog(`!!! ERROR DE CONEXIÓN: ${err.message}`);
        alert(`Error de respuesta: No se pudo comunicar con el clúster. Verifica que la terminal de Python esté encendida. (${err.message})`);
    } finally {
        scanBtn.disabled = false;
        scanBtn.innerText = "EJECUTAR ESCANEO LOCAL";
        document.getElementById('scanner').style.display = 'none';
    }
}

scanBtn.onclick = analyzeFile;

async function analyzeUrl() {
    if (scanBtn.disabled) return;
    clearLogs();
    const urlInput = document.getElementById('urlInput');
    const url = urlInput ? urlInput.value.trim() : '';
    if (!url) {
        alert("TALOS requiere una URL válida.");
        return;
    }

    // Limpiar trackeo y parámetros innecesarios de la URL (FB, IG, etc)
    let cleanUrl = url.split('?')[0];
    if (url.includes('facebook.com/watch')) cleanUrl = url;

    scanBtn.disabled = true;
    const scanUrlBtn = document.getElementById('scanUrlBtn');
    if (scanUrlBtn) scanUrlBtn.disabled = true;

    const isSocial = /instagram|facebook|tiktok|youtube|youtu\.be|twitter|x\.com/.test(url.toLowerCase());

    if (isSocial) {
        const platform = url.includes('instagram') ? 'INSTAGRAM' :
            url.includes('facebook') ? 'FACEBOOK' :
                url.includes('tiktok') ? 'TIKTOK' :
                    url.includes('youtube') ? 'YOUTUBE' : 'RED SOCIAL';

        addLog(`>> DETECTADO ORIGEN: ${platform}`);
        addLog(`>> EXTRACCIÓN SOTA: Desgranado de HTML en curso...`);
        addLog(`>> BYPASSING COOKIES: Consultando bases de datos locales...`);
        scanBtn.innerText = "EXTRAYENDO VIDEO...";
    } else {
        scanBtn.innerText = "DESCARGANDO EVIDENCIA...";
    }

    document.getElementById('scannerUrl').style.display = 'block';
    addLog(`INICIANDO ANÁLISIS RED EXTERNA: ${cleanUrl.substring(0, 50)}...`);

    try {
        const response = await fetch('http://127.0.0.1:8000/analizar-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: cleanUrl })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: Error en el clúster.`);
        }

        const data = await response.json();

        if (data.status === "error") {
            addLog(`!!! ERROR URL: ${data.error}`);
            alert(`Talos rechazó la URL: ${data.error}`);
            return;
        }

        const finalVerdict = data.verdict || (data.probabilidad >= 50 ? 'IA' : 'REAL');
        renderResult(data);
        saveToHistory(url.substring(0, 30) + '...', finalVerdict, data);
    } catch (err) {
        console.error(err);
        addLog(`!!! ERROR DE CONEXIÓN: ${err.message}`);
        alert(`Error: No se pudo comunicar con el clúster remoto. (${err.message})`);
    } finally {
        scanBtn.disabled = false;
        if (scanUrlBtn) scanUrlBtn.disabled = false;
        scanBtn.innerText = "EJECUTAR ESCANEO LOCAL";
        document.getElementById('scannerUrl').style.display = 'none';
        urlInput.value = '';
    }
}

const scanUrlBtnInit = document.getElementById('scanUrlBtn');
if (scanUrlBtnInit) scanUrlBtnInit.onclick = analyzeUrl;

window.handleUrlKeyPress = function (e) {
    if (e.key === 'Enter') {
        analyzeUrl();
    }
};

/* ----- URL PREVIEW (WhatsApp style) ----- */
let _urlPreviewTimer = null;

window.clearUrlInput = function (e) {
    if (e) {
        e.preventDefault();
        e.stopPropagation();
    }
    const urlInput = document.getElementById('urlInput');
    if (urlInput) {
        urlInput.value = '';
        urlInput.focus();
        window.handleUrlPreview('');
    }
};

window.handleUrlPreview = function (rawVal) {
    const val = rawVal.trim();
    const card = document.getElementById('urlPreviewCard');
    const loading = document.getElementById('urlPreviewLoading');
    const content = document.getElementById('urlPreviewContent');
    const header = document.getElementById('urlScanHeader');

    const clearUrlBtn = document.getElementById('clearUrlBtn');

    // Si está vacío, ocultar preview y mostrar header de nuevo
    if (!val) {
        if (card) card.classList.add('d-none');
        if (clearUrlBtn) clearUrlBtn.style.display = 'none';
        if (header) {
            header.style.maxHeight = '300px';
            header.style.opacity = '1';
            header.style.marginBottom = '';
            header.style.pointerEvents = '';
            header.style.overflow = '';
        }
        return;
    }

    // Hay contenido: mostramos el botón de borrar (aunque la url no sea aún válida del todo)
    if (clearUrlBtn) clearUrlBtn.style.display = 'flex';

    // Validar que sea una URL mínimamente válida
    let parsedUrl = null;
    try {
        parsedUrl = new URL(val.startsWith('http') ? val : 'https://' + val);
    } catch (_) {
        if (card) card.classList.add('d-none');
        if (header) {
            header.style.maxHeight = '300px';
            header.style.opacity = '1';
        }
        return;
    }

    // Ocultar el header con transición suave
    if (header) {
        header.style.transition = 'max-height 0.35s ease, opacity 0.3s ease, margin 0.3s ease';
        header.style.maxHeight = '0px';
        header.style.opacity = '0';
        header.style.marginBottom = '0';
        header.style.overflow = 'hidden';
        header.style.pointerEvents = 'none';
    }

    // Mostrar card con estado loading
    card.classList.remove('d-none');
    loading.classList.remove('d-none');
    content.classList.add('d-none');

    // Debounce
    clearTimeout(_urlPreviewTimer);
    _urlPreviewTimer = setTimeout(() => {
        _buildUrlPreview(val, parsedUrl);
    }, 500);
};

function _buildUrlPreview(url, parsed) {
    const hostname = parsed.hostname.replace('www.', '');
    const path = parsed.pathname;

    // ── Detectar plataforma & tipo ────────────────────────────────
    let thumbSrc = null;
    let thumbIconClass = null;
    let thumbIconColor = '#00ffcc';
    let badgeIcon = 'fa-link';
    let badgeText = 'ENLACE';
    let badgeColor = '#00ffcc';
    let title = 'Contenido externo';
    let meta = 'Analizar con TALOS Forensic';
    let fetchRemoteThumb = false; // si debe buscar miniatura con Microlink

    const lcu = url.toLowerCase();
    const isYoutube = /youtube\.com|youtu\.be/.test(lcu);
    const isTikTok = /tiktok\.com/.test(lcu);
    const isInstagram = /instagram\.com/.test(lcu);
    const isFacebook = /facebook\.com|fb\.watch/.test(lcu);
    const isTwitter = /twitter\.com|x\.com/.test(lcu);
    const isDirectImage = /\.(jpg|jpeg|png|gif|webp|bmp|svg)(\?|#|$)/i.test(path);
    const isDirectVideo = /\.(mp4|webm|avi|mov|mkv|m4v)(\?|#|$)/i.test(path);
    const isDirectAudio = /\.(mp3|wav|ogg|flac|aac|m4a|opus)(\?|#|$)/i.test(path);

    if (isYoutube) {
        let videoId = null;
        const ytMatch = url.match(/(?:v=|youtu\.be\/|embed\/|shorts\/)([A-Za-z0-9_-]{11})/);
        if (ytMatch) videoId = ytMatch[1];

        thumbSrc = videoId ? `https://img.youtube.com/vi/${videoId}/hqdefault.jpg` : null;
        thumbIconClass = 'fa-brands fa-youtube';
        thumbIconColor = '#FF0000';
        badgeIcon = 'fa-brands fa-youtube';
        badgeText = 'YOUTUBE';
        badgeColor = '#FF0000';
        title = videoId ? `Video de YouTube` : 'Canal / Playlist de YouTube';
        meta = videoId ? `ID: ${videoId} · Video detectado · Listo para extracción forense` : 'Perfil o playlist · Extracción por yt-dlp';
    } else if (isTikTok) {
        thumbIconClass = 'fa-brands fa-tiktok';
        thumbIconColor = '#69C9D0';
        badgeIcon = 'fa-brands fa-tiktok';
        badgeText = 'TIKTOK';
        badgeColor = '#69C9D0';
        title = 'Video de TikTok';
        meta = 'Extracción mediante motor SOTA · Bypass de cookies activo';
        fetchRemoteThumb = true;
    } else if (isInstagram) {
        thumbIconClass = 'fa-brands fa-instagram';
        thumbIconColor = '#E1306C';
        badgeIcon = 'fa-brands fa-instagram';
        badgeText = 'INSTAGRAM';
        badgeColor = '#E1306C';
        title = path.includes('/reel') ? 'Reel de Instagram' : 'Post de Instagram';
        meta = 'Extracción de media privada · Análisis forense de compresión';
        fetchRemoteThumb = true;
    } else if (isFacebook) {
        thumbIconClass = 'fa-brands fa-facebook';
        thumbIconColor = '#1877F2';
        badgeIcon = 'fa-brands fa-facebook';
        badgeText = 'FACEBOOK';
        badgeColor = '#1877F2';
        title = 'Video de Facebook';
        meta = 'Extracción mediante parser HTML · Desgranado de CDN activo';
        fetchRemoteThumb = true;
    } else if (isTwitter) {
        thumbIconClass = 'fa-brands fa-x-twitter';
        thumbIconColor = '#ffffff';
        badgeIcon = 'fa-brands fa-x-twitter';
        badgeText = 'X / TWITTER';
        badgeColor = '#ffffff';
        title = 'Post de X / Twitter';
        meta = 'Media tweet detectada · Listo para análisis de deepfake';
        fetchRemoteThumb = true;
    } else if (isDirectImage) {
        thumbSrc = url;
        badgeIcon = 'fa-solid fa-image';
        badgeText = 'IMAGEN';
        badgeColor = '#00ffcc';
        title = path.split('/').pop() || 'Imagen remota';
        meta = `Imagen directa · Host: ${hostname}`;
    } else if (isDirectVideo) {
        thumbIconClass = 'fa-solid fa-film';
        thumbIconColor = '#ff0055';
        badgeIcon = 'fa-solid fa-film';
        badgeText = 'VIDEO';
        badgeColor = '#ff0055';
        title = path.split('/').pop() || 'Video remoto';
        meta = `Archivo de video · Host: ${hostname} · Análisis deepfake activado`;
        fetchRemoteThumb = true;
    } else if (isDirectAudio) {
        thumbIconClass = 'fa-solid fa-waveform-lines';
        thumbIconColor = '#7c5cbf';
        badgeIcon = 'fa-solid fa-waveform-lines';
        badgeText = 'AUDIO';
        badgeColor = '#7c5cbf';
        title = path.split('/').pop() || 'Audio remoto';
        meta = `Archivo de audio · Host: ${hostname} · Análisis de voz sintética`;
    } else {
        thumbIconClass = 'fa-solid fa-globe';
        thumbIconColor = '#888';
        badgeIcon = 'fa-solid fa-globe';
        badgeText = 'ENLACE';
        badgeColor = '#aaa';
        title = hostname;
        meta = `URL genérica · ${hostname} · TALOS intentará extracción`;
        fetchRemoteThumb = true;
    }

    // ── Aplicar al DOM ────────────────────────────────────────────
    const loading = document.getElementById('urlPreviewLoading');
    const contentEl = document.getElementById('urlPreviewContent');
    const thumbImg = document.getElementById('urlThumbImg');
    const thumbIcon = document.getElementById('urlThumbIcon');
    const thumbIconEl = document.getElementById('urlThumbIconEl');
    const badge = document.getElementById('urlPreviewBadge');
    const badgeIconEl = document.getElementById('urlPreviewBadgeIcon');
    const badgeTextEl = document.getElementById('urlPreviewBadgeText');
    const domainEl = document.getElementById('urlPreviewDomain');
    const titleEl = document.getElementById('urlPreviewTitle');
    const metaEl = document.getElementById('urlPreviewMeta');

    // Badge
    badgeIconEl.className = `${badgeIcon} me-1`;
    badgeTextEl.innerText = badgeText;
    badge.style.color = badgeColor;
    badge.style.borderColor = badgeColor + '55';
    badge.style.backgroundColor = badgeColor + '15';

    // Dominio
    domainEl.innerText = hostname;

    // Título y meta
    titleEl.innerText = title;
    metaEl.innerText = meta;

    // Thumbnail inicial: imagen directa o ícono de plataforma
    thumbImg.classList.add('d-none');
    thumbIcon.classList.add('d-none');

    const _showIcon = () => {
        thumbIcon.classList.remove('d-none');
        thumbIconEl.className = thumbIconClass || 'fa-solid fa-globe';
        thumbIcon.style.color = thumbIconColor;
    };

    const _showImage = (src) => {
        thumbImg.classList.remove('d-none');
        thumbImg.style.opacity = '0';
        thumbImg.onload = () => { thumbImg.style.opacity = '1'; };
        thumbImg.onerror = () => {
            thumbImg.classList.add('d-none');
            _showIcon();
        };
        thumbImg.src = src;
    };

    if (thumbSrc) {
        _showImage(thumbSrc);
    } else if (thumbIconClass) {
        _showIcon();
    }

    // Mostrar content, ocultar loading
    loading.classList.add('d-none');
    contentEl.classList.remove('d-none');
    contentEl.style.animation = 'none';
    contentEl.offsetHeight; // reflow
    contentEl.style.animation = '';

    if (fetchRemoteThumb) {
        // TikTok tiene oEmbed oficial → más fiable que Microlink
        if (/tiktok\.com/.test(url.toLowerCase())) {
            _fetchTikTokThumb(url, { titleEl, metaEl, thumbImg, thumbIcon, thumbIconEl, thumbIconColor, thumbIconClass, _showImage, _showIcon });
        } else {
            _fetchMicriolinkThumb(url, { titleEl, metaEl, thumbImg, thumbIcon, thumbIconEl, thumbIconColor, thumbIconClass, _showImage, _showIcon });
        }
    }
}

// ── TikTok oEmbed (API oficial, no requiere auth) ─────────────────────────
async function _fetchTikTokThumb(url, refs) {
    try {
        const oembedUrl = `https://www.tiktok.com/oembed?url=${encodeURIComponent(url)}`;
        const res = await fetch(oembedUrl, { signal: AbortSignal.timeout(7000) });
        if (!res.ok) throw new Error('oembed_fail');
        const data = await res.json();

        const thumbUrl   = data.thumbnail_url || null;
        const videoTitle = data.title         || null;
        const author     = data.author_name   || null;

        if (videoTitle) refs.titleEl.innerText = videoTitle;
        if (author)     refs.metaEl.innerText  = `@${author} · TikTok · Listo para análisis forense`;

        if (thumbUrl) {
            refs.thumbImg.classList.add('d-none');
            refs.thumbIcon.classList.add('d-none');
            refs._showImage(thumbUrl);
        }
    } catch (_) {
        // Fallback a Microlink si oEmbed falló
        _fetchMicriolinkThumb(url, refs);
    }
}

async function _fetchMicriolinkThumb(url, refs) {
    try {
        const apiUrl = `https://api.microlink.io?url=${encodeURIComponent(url)}&screenshot=false&meta=true`;
        const res = await fetch(apiUrl, { signal: AbortSignal.timeout(6000) });
        if (!res.ok) return;
        const data = await res.json();
        if (data.status !== 'success') return;

        const ogImage = data.data?.image?.url || data.data?.logo?.url || null;
        const ogTitle = data.data?.title || null;
        const ogDesc  = data.data?.description || null;

        // Actualizar título si lo tenemos
        if (ogTitle) refs.titleEl.innerText = ogTitle;
        if (ogDesc)  refs.metaEl.innerText  = ogDesc.substring(0, 80) + (ogDesc.length > 80 ? '…' : '');

        // Actualizar thumbnail si hay imagen OG
        if (ogImage) {
            refs.thumbImg.classList.add('d-none');
            refs.thumbIcon.classList.add('d-none');
            refs._showImage(ogImage);
        }
    } catch (_) {
        // Silencioso — el ícono de plataforma ya está mostrado
    }
}



function renderResult(data) {
    const prob = parseFloat(data.probabilidad) || 0;

    // ── Debug forense completo en consola ──────────────────────────
    console.group(`%c[TALOS DEBUG] Resultado: ${prob}% → ${data.verdict}`,
        `color: ${prob > 69 ? '#ff0055' : prob > 40 ? '#ffcc00' : '#00ffcc'}; font-weight: bold; font-size: 13px`);
    console.log('📊 Probabilidad:', prob + '%');
    console.log('⚖️ Veredicto backend:', data.verdict);
    console.log('📡 Tipo:', data.tipo);
    if (data.module_scores) {
        console.log('🔬 Scores por módulo:', data.module_scores);
    }
    if (data.forensic_report) {
        console.log('🧬 Reporte forense:', data.forensic_report);
    }
    if (data.reasons) {
        console.log('📝 Razones:', data.reasons);
    }
    console.groupEnd();

    addLog(`[VEREDICTO] Prob=${prob}% | Módulos: ${JSON.stringify(data.module_scores || {})}`);

    const uiUpload = document.getElementById('ui-upload');
    const uiResult = document.getElementById('ui-result');

    if (uiUpload) uiUpload.classList.add('d-none');
    if (uiResult) uiResult.classList.remove('d-none');

    // Normalizar veredicto solo como fallback si no hay probabilidad
    let verdictStr = (data.verdict || "").toUpperCase();
    if (!verdictStr) {
        verdictStr = prob >= 50 ? "IA" : "REAL";
    }

    // Actualizar Subtítulo dinámico según previsualización activa
    const subtitle = document.getElementById('analysis-type-subtitle');
    if (subtitle) {
        const isVideo = !document.getElementById('videoPreview').classList.contains('d-none');
        const isImage = !document.getElementById('imgPreview').classList.contains('d-none');
        const isAudio = document.getElementById('audioWrapper').style.display === 'block';

        if (isVideo) {
            subtitle.innerText = "Resultados del análisis de video";
        } else if (isImage) {
            subtitle.innerText = "Resultados del análisis de imagen";
        } else if (isAudio) {
            subtitle.innerText = "Resultados del análisis de audio";
        } else {
            subtitle.innerText = "Resultados del análisis de evidencia";
        }
    }

    // Actualizar Caja de Veredicto (Referencia)
    const vTitle = document.getElementById('verdict-title');
    const vDesc = document.getElementById('verdict-description');
    const vIconMain = document.getElementById('verdict-icon-main');
    const vIconAlert = document.getElementById('verdict-icon-alert');
    const vFooter = document.getElementById('verdict-footer');

    const mainPanel = document.getElementById('mainPanel');

    // Resetear clases y estados
    mainPanel.classList.remove('result-ia', 'result-real', 'result-warn');

    // Determinar estado final basado en probabilidad (MODELO BALANCEADO: 0-40 Real, 41-69 Incierto, 70-100 IA)
    let finalStatus = "";
    if (data.probabilidad !== undefined && data.probabilidad !== null && data.probabilidad !== "") {
        if (prob <= 40) {
            finalStatus = "REAL";
        } else if (prob <= 69) {
            finalStatus = "INCIERTO";
        } else {
            finalStatus = "IA";
        }
    } else {
        // Fallback si no hay prob
        const isSintetico = ["IA", "SINTÉTICO", "SINTETICO", "FAKE", "MANIPULADO"].includes(verdictStr);
        finalStatus = isSintetico ? "IA" : "REAL";
    }

    if (finalStatus === "IA") {
        mainPanel.classList.add('result-ia');
        vTitle.innerText = "Este contenido fue creado con inteligencia artificial";
        vDesc.innerText = "Tras el análisis de micro-estructuras térmicas y coherencia de ruido espectral, se han identificado señales claras de intervención con IA. Este contenido no fue producido de manera completamente real.";
        vIconMain.innerHTML = '<i class="fa-solid fa-robot"></i>';
        vIconAlert.innerHTML = '<i class="fa-solid fa-bell"></i>';
        vFooter.innerText = "Te recomendamos hacer una verificación adicional y revisar la fuente original antes de compartirlo.";
    } else if (finalStatus === "INCIERTO") {
        mainPanel.classList.add('result-warn');
        vTitle.innerText = "Análisis de posible intervención con inteligencia artificial";
        vDesc.innerText = "Las herramientas no lograron determinar si el contenido fue creado o no con IA. Puede deberse a la calidad o a la complejidad del archivo analizado.";
        vIconMain.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
        vIconAlert.innerHTML = '';

        let footerText = "En estos casos, te invitamos a revisar el origen del contenido o buscar la fuente original para confirmar su autenticidad.";
        if (prob >= 55) {
            vFooter.innerText = "Alta sospecha, pero no confirmado. " + footerText;
        } else {
            vFooter.innerText = "Resultado poco concluyente. " + footerText;
        }
    } else {
        mainPanel.classList.add('result-real');
        vTitle.innerText = "Este contenido ha sido validado satisfactoriamente";
        vDesc.innerText = "La integridad de la muestra ha sido verificada. Los patrones de distribución de fotones coinciden con una captura física fidedigna de hardware óptico. Sin rastro de manipulación sintética detectada.";
        vIconMain.innerHTML = '<i class="fa-solid fa-user-check"></i>';
        vIconAlert.innerHTML = '<i class="fa-solid fa-shield-halved"></i>';
        vFooter.innerText = "Este contenido ha pasado los protocolos de validación TALOS de captura orgánica.";
    }
}



/* ----- HISTORIAL Y NAVEGACIÓN ----- */
window.switchTab = function (tab) {
    const scanPanel = document.getElementById('scanPanel');
    const dataPanel = document.getElementById('dataPanel');
    const uiUpload = document.getElementById('ui-upload');
    const uiResult = document.getElementById('ui-result');
    const uiDb = document.getElementById('ui-database');
    const uiCont = document.getElementById('ui-contact');
    const uiAcademy = document.getElementById('ui-academy');
    const uiChallenge = document.getElementById('ui-challenge');
    const helpBtn = document.getElementById('btnHelp');

    document.querySelectorAll('.nav-menu li').forEach(li => li.classList.remove('active'));

    if (tab === 'scanner') {
        document.getElementById('nav-scanner').classList.add('active');
        scanPanel.classList.remove('d-none');
        dataPanel.classList.add('d-none');
        uiResult.classList.add('d-none');
        uiUpload.classList.remove('d-none');

        // Limpieza de estados de resultado
        const mainPanel = document.getElementById('mainPanel');
        if (mainPanel) mainPanel.classList.remove('result-ia', 'result-real', 'result-warn');

        // Limpieza completa y reset a la vista de botones
        if (typeof window.resetScanMode === 'function') {
            window.resetScanMode();
        } else {
            addLog("SISTEMA RESETEADO. SELECCIONE MODO DE ESCANEO...");
        }
    } else {
        scanPanel.classList.add('d-none');
        dataPanel.classList.remove('d-none');
        uiDb.classList.add('d-none');
        if (uiCont) uiCont.classList.add('d-none');
        if (uiAcademy) uiAcademy.classList.add('d-none');
        if (uiChallenge) uiChallenge.classList.add('d-none');

        if (tab === 'database') {
            document.getElementById('nav-database').classList.add('active');
            uiDb.classList.remove('d-none');
            loadDatabase();
        } else if (tab === 'contact') {
            document.getElementById('nav-contact').classList.add('active');
            uiCont.classList.remove('d-none');
        } else if (tab === 'academy') {
            document.getElementById('nav-academy').classList.add('active');
            uiAcademy.classList.remove('d-none');
        } else if (tab === 'challenge') {
            document.getElementById('nav-challenge').classList.add('active');
            uiChallenge.classList.remove('d-none');
        }
    }
};

function saveToHistory(name, verdict, details) {
    const history = JSON.parse(localStorage.getItem('talos_history') || '[]');
    let finalVerdict = verdict;
    if (details && details.probabilidad !== undefined && details.probabilidad !== null && details.probabilidad !== "") {
        let prob = parseFloat(details.probabilidad) || 0;
        if (prob <= 40) finalVerdict = "REAL";
        else if (prob <= 69) finalVerdict = "INCIERTO";
        else finalVerdict = "IA";
    }
    history.unshift({ name, verdict: finalVerdict || 'REAL', details, date: new Date().toLocaleString() });
    localStorage.setItem('talos_history', JSON.stringify(history.slice(0, 50)));
}

function loadDatabase() {
    const history = JSON.parse(localStorage.getItem('talos_history') || '[]');
    const body = document.getElementById('dbTableBody');
    if (!body) return;
    body.innerHTML = '';

    if (history.length === 0) {
        body.innerHTML = `<tr><td colspan="4" class="text-center p-5 text-secondary opacity-50">NO SE ENCONTRARON REGISTROS EN EL CLÚSTER LOCAL</td></tr>`;
    }

    history.forEach((item, index) => {
        const tr = document.createElement('tr');
        let probInfo = item.details && item.details.probabilidad !== undefined && item.details.probabilidad !== null && item.details.probabilidad !== "" ? parseFloat(item.details.probabilidad) : null;
        let displayVerdict = item.verdict || "";
        if (probInfo !== null) {
            if (probInfo <= 40) displayVerdict = "REAL";
            else if (probInfo <= 69) displayVerdict = "INCIERTO";
            else displayVerdict = "IA";
        }
        displayVerdict = displayVerdict.toUpperCase();

        let badgeClass = 'real';
        let badgeText = 'ORGÁNICO';
        let badgeIcon = '<i class="fa-solid fa-user-check"></i>';

        if (["IA", "SINTÉTICO", "SINTETICO", "FAKE", "MANIPULADO"].includes(displayVerdict)) {
            badgeClass = 'ia';
            badgeText = 'SINTÉTICO';
            badgeIcon = '<i class="fa-solid fa-robot"></i>';
        } else if (displayVerdict === "INCIERTO") {
            badgeClass = 'warn';
            badgeText = 'INCIERTO';
            badgeIcon = '<i class="fa-solid fa-triangle-exclamation"></i>';
        }

        // Formatear fecha para que se vea más limpia
        const dateParts = item.date.split(', ');
        const dateStr = dateParts[0];
        const timeStr = dateParts[1] || "";

        tr.innerHTML = `
            <td>
                <div class="d-flex flex-column">
                    <span class="text-white small">${dateStr}</span>
                    <span class="extra-small text-muted">${timeStr}</span>
                </div>
            </td>
            <td>
                <div class="d-flex align-items-center gap-2">
                    <i class="fa-solid fa-file-lines text-muted" style="font-size: 0.9rem;"></i>
                    <span class="fw-bold extra-small" style="color: #cbd5e0;">${item.name}</span>
                </div>
            </td>
            <td>
                <span class="verdict-badge ${badgeClass}">
                    ${badgeIcon} ${badgeText}
                </span>
            </td>
            <td class="text-end">
                <button class="btn btn-sm btn-outline-info extra-small fw-bold px-3" onclick="generatePDF(${index})" style="border-radius: 4px; letter-spacing: 1px;">
                    <i class="fa-solid fa-file-pdf me-1"></i> REPORTE PDF
                </button>
            </td>
        `;
        body.appendChild(tr);
    });
    const stats = document.getElementById('dbStats');
    if (stats) stats.innerText = `REGISTROS: ${history.length}`;
}

// Cerrar sidebar al navegar en móviles
document.querySelectorAll('.nav-menu a').forEach(link => {
    link.addEventListener('click', () => {
        if (window.innerWidth <= 991) {
            sidebar.classList.remove('open');
            sidebar.classList.add('collapsed');
            document.body.classList.add('sidebar-collapsed');
            if (sidebarBackdrop) sidebarBackdrop.classList.remove('active');
        }
    });
});

window.clearDatabase = function () {
    if (confirm("¿Seguro que deseas purgar todos los registros forenses?")) {
        localStorage.removeItem('talos_history');
        loadDatabase();
    }
};

window.reprintLog = function (index) {
    const history = JSON.parse(localStorage.getItem('talos_history') || '[]');
    renderResult(history[index].details);
};

/* ----- SIMULADOR / EXAMEN IA ----- */
let challengeScore = 0;
let currentChallenge = 0;
let canAnswer = true;

const allChallengeData = [{"type": "audio", "src": "backend/data/audio 1.mp3", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/audio 2.mp3", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/audio 3.mp3", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/audio 4.mp3", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/audio 5.mp3", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/audio ia 1.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 10.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 11.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 12.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 13.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 14.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 15.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 16.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 2.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 3.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 4.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 5.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 6.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 7.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 8.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "audio", "src": "backend/data/audio ia 9.mp3", "ans": "IA", "reason": "Se detectan inconsistencias y artefactos espectrales."}, {"type": "image", "src": "backend/data/Captura de pantalla 2026-04-13 234338.png", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/Donald Trump vs Vladimir Putin in UFC Fight.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/foto ia 10.png", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 11.jpg", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 12.jpg", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 2.png", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 3.jpg", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 4.png", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 5.jpg", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 6.jpg", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 7.png", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 8.png", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto ia 9.png", "ans": "IA", "reason": "Se detectan inconsistencias de borde e iluminación neuronal."}, {"type": "image", "src": "backend/data/foto real 2.jpg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/foto real 4.jpg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/foto real 5.jpg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/foto real 6.jpg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/Image-removebg-preview.png", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/josuefarias32 - 7379077518009126149.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "image", "src": "backend/data/Pi7_Passport_Photo.jpeg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/tacometro de velocidad.png", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video ia 10.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia 2.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia 3.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia 4.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia 5.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia 6.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia 7.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia 8.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia 9.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video ia.mp4", "ans": "IA", "reason": "Se detectan inconsistencias de coherencia temporal generativa."}, {"type": "video", "src": "backend/data/video instagram prueba.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 10.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 2.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 3.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 4.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 5.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 6.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 7.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 8.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real 9.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/video real.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/video tik tok prueba.mp3", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/WhatsApp Image 2026-03-26 at 11.19.12.jpeg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/WhatsApp Image 2026-03-30 at 21.25.01 (1).jpeg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/WhatsApp Image 2026-03-31 at 00.00.55 (1).jpeg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/WhatsApp Image 2026-03-31 at 00.00.55 (2).jpeg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/WhatsApp Image 2026-03-31 at 00.00.55.jpeg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/WhatsApp Image 2026-04-10 at 14.52.08.jpeg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/WhatsApp Ptt 2026-03-30 at 21.29.32.ogg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/WhatsApp Ptt 2026-04-10 at 17.28.20.ogg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/WhatsApp Ptt 2026-04-13 at 21.39.02.ogg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "audio", "src": "backend/data/WhatsApp Ptt 2026-04-13 at 21.58.03.ogg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/WhatsApp Video 2026-04-06 at 22.06.47.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "video", "src": "backend/data/WhatsApp Video 2026-04-09 at 22.37.29.mp4", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}, {"type": "image", "src": "backend/data/WIN_20260317_22_04_57_Pro.jpg", "ans": "REAL", "reason": "Coherencia física total y ausencia de artefactos sintéticos."}];
let challengeData = [];

window.startChallenge = function () {
    challengeScore = 0;
    currentChallenge = 0;
    
    // Seleccionar 10 elementos aleatorios en cada intento de uso
    challengeData = [...allChallengeData].sort(() => 0.5 - Math.random()).slice(0, 10);
    
    document.getElementById('challenge-intro').classList.add('d-none');
    document.getElementById('challenge-result').classList.add('d-none');
    document.getElementById('challenge-game').classList.remove('d-none');
    showChallenge();
};

function showChallenge() {
    const item = challengeData[currentChallenge];
    document.getElementById('challenge-current').innerText = currentChallenge + 1;
    const media = document.getElementById('challenge-media');
    media.innerHTML = '';

    if (item.type === 'image') {
        const img = document.createElement('img');
        img.src = item.src;
        img.style.maxWidth = '100%';
        img.style.maxHeight = '400px';
        img.className = 'img-fluid rounded shadow-lg';
        media.appendChild(img);
    } else if (item.type === 'video') {
        const vid = document.createElement('video');
        vid.src = item.src;
        vid.controls = true;
        vid.style.width = '100%';
        vid.style.maxHeight = '400px';
        vid.className = 'rounded shadow-lg';
        media.appendChild(vid);
    } else if (item.type === 'audio') {
        const aud = document.createElement('audio');
        aud.src = item.src;
        aud.controls = true;
        aud.style.width = '80%';
        aud.style.margin = '40px auto';
        aud.className = 'd-block';
        media.appendChild(aud);
        const p = document.createElement('p');
        p.innerText = "ESCUCHA ATENTAMENTE LA FIRMA ACÚSTICA";
        p.className = 'text-info small mt-2 fw-bold';
        media.appendChild(p);
    }

    document.getElementById('challenge-feedback').classList.add('d-none');
    document.getElementById('challenge-buttons').classList.remove('d-none');
    canAnswer = true;
}

window.answerChallenge = function (ans) {
    if (!canAnswer) return;
    canAnswer = false;

    const item = challengeData[currentChallenge];
    const isCorrect = ans === item.ans;
    if (isCorrect) challengeScore++;

    document.getElementById('challenge-buttons').classList.add('d-none');
    const feedback = document.getElementById('challenge-feedback');
    feedback.classList.remove('d-none');

    const fTitle = document.getElementById('feedback-title');
    fTitle.innerText = isCorrect ? "NODO VALIDADO - CORRECTO" : "FALLO DE DETECCIÓN - INCORRECTO";
    fTitle.className = isCorrect ? "text-success h4 mb-2" : "text-danger h4 mb-2";

    document.getElementById('feedback-reason').innerText = item.reason;
};

window.nextChallenge = function () {
    currentChallenge++;
    if (currentChallenge < challengeData.length) {
        showChallenge();
    } else {
        endChallenge();
    }
};

function endChallenge() {
    document.getElementById('challenge-game').classList.add('d-none');
    document.getElementById('challenge-result').classList.remove('d-none');
    document.getElementById('challenge-score').innerText = `${challengeScore} / ${challengeData.length}`;

    const resTitle = document.querySelector('#challenge-result h2');
    if (challengeScore === challengeData.length) {
        resTitle.innerText = "RANGO: EXPERTO FORENSE";
        resTitle.className = "text-info mb-2 fw-bold";
    } else if (challengeScore >= 7) {
        resTitle.innerText = "RANGO: ANALISTA AVANZADO";
        resTitle.className = "text-success mb-2 fw-bold";
    } else {
        resTitle.innerText = "RANGO: REENTRENAMIENTO REQUERIDO";
        resTitle.className = "text-warning mb-2 fw-bold";
    }
}

window.resetChallenge = () => startChallenge();

/* ----- BACKGROUND PARTICLES (GEOMETRIC SHAPES) ----- */
const canvas = document.getElementById('bgCanvas');
if (canvas) {
    const ctx = canvas.getContext('2d');
    let particles = [];
    const mouse = { x: null, y: null, radius: 150 };

    window.addEventListener('mousemove', (e) => {
        mouse.x = e.x;
        mouse.y = e.y;
    });

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 1.5;
            this.vy = (Math.random() - 0.5) * 1.5;
            this.size = Math.random() * 2 + 1;
            this.density = (Math.random() * 30) + 10;
            this.color = Math.random() > 0.5 ? '#ff0055' : '#00ffcc';
        }

        draw() {
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }

        update() {
            // Movimiento autónomo (Drift)
            this.x += this.vx;
            this.y += this.vy;

            // Rebote en bordes
            if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height) this.vy *= -1;

            // Evasión de cursor
            let dx = mouse.x - this.x;
            let dy = mouse.y - this.y;
            let distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < mouse.radius) {
                let forceDirectionX = dx / distance;
                let forceDirectionY = dy / distance;
                let force = (mouse.radius - distance) / mouse.radius;
                let directionX = forceDirectionX * force * 5;
                let directionY = forceDirectionY * force * 5;
                this.x -= directionX;
                this.y -= directionY;
            }
        }
    }

    function init() {
        particles = [];
        const numberOfParticles = (canvas.width * canvas.height) / 9000;
        for (let i = 0; i < numberOfParticles; i++) {
            particles.push(new Particle());
        }
    }

    function connect() {
        let opacityValue = 1;
        for (let a = 0; a < particles.length; a++) {
            for (let b = a; b < particles.length; b++) {
                let distance = ((particles[a].x - particles[b].x) * (particles[a].x - particles[b].x))
                    + ((particles[a].y - particles[b].y) * (particles[a].y - particles[b].y));
                if (distance < (canvas.width / 7) * (canvas.height / 7)) {
                    opacityValue = 1 - (distance / 20000);
                    ctx.strokeStyle = `rgba(0, 255, 204, ${opacityValue * 0.2})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(particles[a].x, particles[a].y);
                    ctx.lineTo(particles[b].x, particles[b].y);
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particles.length; i++) {
            particles[i].draw();
            particles[i].update();
        }
        connect();
        requestAnimationFrame(animate);
    }

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        init();
    });

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    init();
    animate();
}

/* ----- PDF GENERATION ----- */
window.generatePDF = function (index) {
    const history = JSON.parse(localStorage.getItem('talos_history') || '[]');
    const item = history[index];
    if (!item) return;

    // Usaremos la biblioteca html2pdf si está disponible, sino simulamos un aviso
    if (typeof html2pdf !== 'undefined') {
        const element = document.createElement('div');
        element.style.padding = '40px';
        element.style.background = '#0a0a0f';
        element.style.color = '#fff';
        element.style.fontFamily = 'monospace';

        element.innerHTML = `
            <h1 style="color: #ff0055; border-bottom: 2px solid #ff0055;">REPORTE FORENSE DE SEGURIDAD</h1>
            <p><strong>ID DE ANÁLISIS:</strong> ${Math.random().toString(36).substr(2, 9).toUpperCase()}</p>
            <p><strong>FECHA:</strong> ${item.date}</p>
            <p><strong>ARCHIVO:</strong> ${item.name}</p>
            <hr style="border: 0; border-top: 1px solid #333;">
            <h2 style="color: #00ffcc;">VEREDICTO: ${item.verdict === 'IA' ? 'SINTÉTICO / IA' : 'HUMANO / REAL'}</h2>
            <p>Probabilidad de IA: ${item.details.probabilidad}%</p>
            
            <p><strong>LOG DE ANÁLISIS COMPLETO:</strong></p>
            <div style="background: #111; padding: 15px; border-left: 3px solid #ff0055; font-size: 0.75rem; color: #aaa; line-height: 1.5;">
                ${item.details.nota || "Sin notas adicionales."}
            </div>

            <p style="margin-top: 30px;"><strong>Detalles Técnicos Estructurados:</strong></p>
            <pre style="background: #111; padding: 10px; border-left: 3px solid #00ffcc; font-size: 0.7rem;">${JSON.stringify(item.details.detalles || {}, null, 2)}</pre>
            
            <p style="margin-top: 50px; font-size: 0.8rem; border-top: 1px solid #333; padding-top: 10px;">
                Este documento es una verificación digital generada por el sistema TALOS Forensic V10.3.
            </p>
        `;

        const opt = {
            margin: 1,
            filename: `Talos_Report_${item.name}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2, backgroundColor: '#0a0a0f' },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
        };

        html2pdf().set(opt).from(element).save();
    } else {
        alert("Generando PDF... (Simulado: Instala html2pdf.js para funcionalidad real)");
        // Podríamos imprimir el log como texto por ahora
        reprintLog(index);
    }
};

/* ----- DYNAMIC HUD BITS ----- */
function createHudBit() {
    const container = document.getElementById('hud-bits-container');
    if (!container) return;

    const bit = document.createElement('div');
    bit.className = 'hud-bit';

    // Contenido técnico aleatorio
    const techStrings = [
        '0x' + Math.random().toString(16).substr(2, 4).toUpperCase(),
        'PORT:' + (Math.floor(Math.random() * 9000) + 1000),
        'LAT:' + (Math.random() * 20).toFixed(2) + 'ms',
        'SIG_CONV:' + (Math.random() * 100).toFixed(1) + '%',
        'NODE_' + Math.floor(Math.random() * 99),
        'TRACE_ACTIVE',
        'BUF_LOADING...',
        'SYS_V10.3'
    ];

    bit.innerText = techStrings[Math.floor(Math.random() * techStrings.length)];
    bit.style.left = Math.random() * 100 + 'vw';
    bit.style.top = Math.random() * 100 + 'vh';

    container.appendChild(bit);

    // Limpiar el elemento después de la animación
    setTimeout(() => bit.remove(), 8000);
}

// Iniciar ráfagas de bits
setInterval(createHudBit, 1500);
for (let i = 0; i < 5; i++) setTimeout(createHudBit, Math.random() * 5000);

/* ----- FOOTER STATUS UPDATES ----- */
function updateFooter() {
    const clock = document.getElementById('footer-clock');
    const ticker = document.getElementById('footer-ticker');
    if (!clock || !ticker) return;

    // Actualizar Reloj (con milisegundos)
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-GB', { hour12: false }) + '.' +
        now.getMilliseconds().toString().padStart(3, '0');
    clock.innerText = timeStr;

    // Actualizar Ticker aleatorio
    if (Math.random() > 0.98) {
        const msgs = [
            'Detección de flujo de datos estable...',
            'Verificando llaves de cifrado AES...',
            'Nodo #402 respondiendo con latencia mínima',
            'Sincronizando base de datos local...',
            'Buffer forense optimizado'
        ];
        ticker.innerText = msgs[Math.floor(Math.random() * msgs.length)];
    }
}

setInterval(updateFooter, 50);

/* ----- LOGOUT FUNCTION ----- */
window.logout = async function() {
    const sessionUser = localStorage.getItem('talos_session') || 'Desconocido';
    try {
        await fetch('http://127.0.0.1:8000/auth/logout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: sessionUser })
        });
    } catch (e) {
        console.error("Fallo al notificar logout al servidor.");
    }
    localStorage.removeItem('talos_session');
    window.location.href = 'login.html';
};
