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
            document.getElementById('helpTitle').innerText = "MANUAL DEL ESCÁNER";
            document.getElementById('help-scanner').classList.remove('d-none');
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
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    terminal.innerHTML += `<br>> [${time}] ${t}`;
    terminal.scrollTop = terminal.scrollHeight;
}

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
    const file = fileInput.files[0];
    if (!file) return;

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
        scanBtn.innerText = "EJECUTAR ESCANEO";
        document.getElementById('scanner').style.display = 'none';
    }
}

scanBtn.onclick = analyzeFile;

function renderResult(data) {
    const uiUpload = document.getElementById('ui-upload');
    const uiResult = document.getElementById('ui-result');

    if (uiUpload) uiUpload.classList.add('d-none');
    if (uiResult) uiResult.classList.remove('d-none');

    const prob = data.probabilidad;
    const verdict = data.verdict || (prob > 50 ? "IA" : "REAL");

    // Resumen inteligente para la interfaz: Extraemos solo lo vital
    const fullNote = data.nota || "";
    let summary = "";
    if (fullNote) {
        const parts = fullNote.split(/\||→|\n/);
        const keywords = ['Firma', 'Clasificador', 'Ensamble', 'Metadata', 'Gemini', 'Grok', 'IA', 'REAL'];
        const matches = parts.filter(p => keywords.some(k => p.includes(k)))
            .map(p => p.trim())
            .filter(p => p.length > 5);

        summary = matches.slice(-2).reverse().join(' • '); // Tomamos los veredictos finales
        if (!summary && fullNote) summary = fullNote.substring(0, 80) + "...";
    }

    document.getElementById('verdict-note').innerText = summary || "ANÁLISIS COMPLETADO";

    const vText = document.getElementById('verdict-text');
    const fill = document.getElementById('gaugePathFill');
    const needleBox = document.getElementById('needleBox');

    // Resetear para forzar animación
    fill.style.strokeDasharray = "0 251.3";
    needleBox.style.transform = "rotate(-90deg)";

    setTimeout(() => {
        const dash = (prob / 100) * 251.3;
        fill.style.strokeDasharray = `${dash} 251.3`;
        needleBox.style.transform = `rotate(${(prob / 100) * 180 - 90}deg)`;

        // Contador fluido usando requestAnimationFrame
        const target = prob;
        const startTime = performance.now();
        const duration = 1500;

        function updateCounter(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing out quadratic
            const easeProgress = 1 - (1 - progress) * (1 - progress);
            const current = Math.floor(easeProgress * target);

            document.getElementById('pct-text').innerText = current + "%";

            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            } else {
                document.getElementById('pct-text').innerText = target + "%";
            }
        }
        requestAnimationFrame(updateCounter);
    }, 50);

    const mainPanel = document.getElementById('mainPanel');
    const iconContainer = document.getElementById('status-icon-container');

    // Resetear clases y estados
    mainPanel.classList.remove('result-ia', 'result-real');
    iconContainer.innerHTML = '';

    if (verdict === "IA" || verdict === "SINTÉTICO") {
        vText.innerText = "CONTENIDO IA DETECTADO";
        vText.style.color = "var(--danger)";
        vText.style.textShadow = "0 0 15px var(--danger)";
        mainPanel.classList.add('result-ia');
        iconContainer.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <path d="M12 9v4"/>
                <path d="M12 17h.01"/>
            </svg>`;
    } else {
        vText.innerText = "CONTENIDO REAL (HUMANO)";
        vText.style.color = "var(--success)";
        vText.style.textShadow = "0 0 15px var(--success)";
        mainPanel.classList.add('result-real');
        iconContainer.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>`;
    }

    // Telemetría
    if (data.detalles && data.detalles.predicciones) {
        document.getElementById('tech-details').classList.remove('d-none');
        const valHuman = data.detalles.predicciones['Humano'] || data.detalles.predicciones['HumanVoice'] || "0%";
        const valAI = data.detalles.predicciones['IA'] || data.detalles.predicciones['AIVoice'] || "0%";

        const labelHuman = document.querySelector('.border-info .text-secondary');
        const labelAI = document.querySelector('.border-danger .text-secondary');

        if (data.tipo === "video") {
            labelHuman.innerText = "VIDEO REAL"; 
            labelAI.innerText = "VIDEO IA";
            
            // Mejora V10: Mostrar sospechoso de Hive si existe
            if (data.forensic_report && data.forensic_report.hive_suspect && data.forensic_report.hive_suspect !== "N/A") {
                const hiveName = data.forensic_report.hive_suspect.toUpperCase();
                const hiveProb = data.forensic_report.hive_ai_prob || 0;
                labelAI.innerText = `IA (${hiveName}: ${hiveProb}%)`;
            }
        } else if (data.tipo === "imagen") {
            labelHuman.innerText = "FOTO REAL"; labelAI.innerText = "FOTO IA";
        } else {
            labelHuman.innerText = "VOZ HUMANA"; labelAI.innerText = "VOZ SINTÉTICA";
        }

        document.getElementById('val-human').innerText = valHuman;
        document.getElementById('val-ai').innerText = valAI;
    } else {
        document.getElementById('tech-details').classList.add('d-none');
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
        if (mainPanel) mainPanel.classList.remove('result-ia', 'result-real');

        // Limpieza completa del estado de carga
        fileInput.value = '';
        const uploadContent = document.getElementById('uploadContent');
        uploadContent.style.setProperty('display', 'flex', 'important');
        uploadContent.classList.remove('d-none');

        const imgPreview = document.getElementById('imgPreview');
        imgPreview.classList.add('d-none');
        imgPreview.src = '';

        const videoPreview = document.getElementById('videoPreview');
        videoPreview.classList.add('d-none');
        videoPreview.src = '';

        const audioWrapper = document.getElementById('audioWrapper');
        audioWrapper.style.display = 'none';
        if (wavesurfer) {
            wavesurfer.destroy();
            wavesurfer = null;
        }

        addLog("SISTEMA RESETEADO. ESPERANDO NUEVA EVIDENCIA...");
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
    const finalVerdict = verdict || (details.probabilidad >= 50 ? 'IA' : 'REAL');
    history.unshift({ name, verdict: finalVerdict, details, date: new Date().toLocaleString() });
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
        const displayVerdict = item.verdict || (item.details && item.details.probabilidad >= 50 ? 'IA' : 'REAL');
        const badgeClass = displayVerdict === 'IA' ? 'text-danger' : 'text-success';
        
        tr.innerHTML = `
            <td>${item.date}</td>
            <td class="fw-bold">${item.name}</td>
            <td><span class="${badgeClass} fw-bold">[ ${displayVerdict} ]</span></td>
            <td class="text-end">
                <button class="btn btn-sm btn-scan" onclick="generatePDF(${index})" style="width: auto; padding: 4px 15px; font-size: 0.6rem;">GENERAR REPORTE</button>
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
const challengeData = [
    { type: 'video', src: 'backend/data/video ia 3.mp4', ans: 'IA', reason: "Se detectan inconsistencias en el flujo óptico y artefactos de borde en el rostro." },
    { type: 'video', src: 'backend/data/video real 2.mp4', ans: 'REAL', reason: "Coherencia física total y ruido térmico de sensor PRNU consistente." },
    { type: 'audio', src: 'backend/data/audio ia 4.mp3', ans: 'IA', reason: "Firma espectral característica de modelos generativos de voz (ElevenLabs)." },
    { type: 'audio', src: 'backend/data/audio 2.mp3', ans: 'REAL', reason: "Presencia de armónicos naturales y variaciones de presión acústica humanas." },
    { type: 'image', src: 'backend/data/foto ia 4.png', ans: 'IA', reason: "Inconsistencia en los reflejos corneales y falta de porosidad natural en la piel." },
    { type: 'image', src: 'backend/data/foto real 2.jpg', ans: 'REAL', reason: "Estructura de píxel Bayer intacta sin patrones de cuadrícula FFT." },
    { type: 'video', src: 'backend/data/video ia.mp4', ans: 'IA', reason: "Desincronización de micro-movimientos musculares (Lip-sync sintético)." },
    { type: 'audio', src: 'backend/data/audio ia 7.mp3', ans: 'IA', reason: "Prosodia monótona y artefactos de fase en las frecuencias altas." },
    { type: 'video', src: 'backend/data/video real.mp4', ans: 'REAL', reason: "Profundidad de campo y desenfoque físico imposible de replicar con precisión actual." },
    { type: 'image', src: 'backend/data/foto ia 5.jpg', ans: 'IA', reason: "Fusión de bordes en áreas de alta frecuencia (cabello) típica de difusión estable." }
];

window.startChallenge = function () {
    challengeScore = 0;
    currentChallenge = 0;
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



