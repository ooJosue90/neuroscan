/* ================================================================
   TALOS ANIME.JS FX MODULE
   Efectos: Cursor Trail, Decode Text, Module Bars,
            Seismic Wave, Glitch RGB, SVG Signature,
            Login Encrypt FX
   ================================================================ */
(function initTalosAnimeFX() {

    /* ── Esperar a que anime.js y el DOM estén listos ─────────── */
    function waitForAnime(cb, tries) {
        tries = tries || 0;
        if (typeof anime !== 'undefined') { cb(); }
        else if (tries < 40) { setTimeout(function () { waitForAnime(cb, tries + 1); }, 100); }
    }

    waitForAnime(function () {

        /* ========================================================
           1. CURSOR TRAIL — estela de puntos cian
        ======================================================== */
        var trailContainer = document.getElementById('cursor-trail-container');
        if (trailContainer) {
            document.addEventListener('mousemove', function (e) {
                var dot = document.createElement('div');
                dot.className = 'cursor-dot';
                dot.style.left = e.clientX + 'px';
                dot.style.top = e.clientY + 'px';
                trailContainer.appendChild(dot);

                anime({
                    targets: dot,
                    opacity: [0.85, 0],
                    scale: [1, 0.1],
                    duration: 480,
                    easing: 'easeOutCubic',
                    complete: function () { if (dot.parentNode) dot.remove(); }
                });
            });
        }

        /* ========================================================
           2. SIDEBAR DECODE TEXT — efecto matrix al abrir menú
        ======================================================== */
        var CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@!%&*';

        function decodeText(el, originalText) {
            var frame = 0;
            var totalFrames = 20;
            var tid = setInterval(function () {
                frame++;
                el.textContent = originalText.split('').map(function (char, i) {
                    if (frame / totalFrames > i / originalText.length) return char;
                    if (char === ' ') return ' ';
                    return CHARSET[Math.floor(Math.random() * CHARSET.length)];
                }).join('');
                if (frame >= totalFrames) {
                    clearInterval(tid);
                    el.textContent = originalText;
                }
            }, 38);
        }

        var sidebarBtn = document.getElementById('sidebarToggle');
        if (sidebarBtn) {
            sidebarBtn.addEventListener('click', function () {
                setTimeout(function () {
                    var navLinks = document.querySelectorAll('.nav-menu a');
                    navLinks.forEach(function (a, idx) {
                        anime({
                            targets: a,
                            translateX: [-24, 0],
                            opacity: [0, 1],
                            delay: idx * 55,
                            duration: 380,
                            easing: 'easeOutExpo'
                        });

                        setTimeout(function () {
                            for (var j = 0; j < a.childNodes.length; j++) {
                                var node = a.childNodes[j];
                                if (node.nodeType === 3 && node.textContent.trim().length > 0) {
                                    var span = document.createElement('span');
                                    var orig = node.textContent.trim();
                                    span.textContent = orig;
                                    a.replaceChild(span, node);
                                    decodeText(span, orig);
                                    break;
                                }
                            }
                        }, idx * 55 + 80);
                    });
                }, 80);
            });
        }

        /* ========================================================
           5. GLITCH RGB — aparece en veredicto IA
        ======================================================== */
        window.talosGlitchVerdict = function (isIA) {
            if (!isIA) return;
            var el = document.getElementById('verdict-title');
            if (!el) return;
            el.classList.remove('glitch-rgb');
            void el.offsetWidth; // reflow
            el.classList.add('glitch-rgb');
            setTimeout(function () { el.classList.remove('glitch-rgb'); }, 1500);
        };

        /* ========================================================
           6. FIRMA SVG DIGITAL — se dibuja al mostrar resultado
        ======================================================== */
        window.talosAnimateSignature = function () {
            var wrap = document.getElementById('svgSignatureWrap');
            var path = document.getElementById('signaturePath');
            if (!wrap || !path) return;
            wrap.classList.add('visible');
            path.style.strokeDashoffset = '400';
            anime({
                targets: path,
                strokeDashoffset: [400, 0],
                duration: 1700,
                delay: 250,
                easing: 'easeInOutCubic'
            });
        };

        window.talosResetSignature = function () {
            var wrap = document.getElementById('svgSignatureWrap');
            var path = document.getElementById('signaturePath');
            if (wrap) wrap.classList.remove('visible');
            if (path) path.style.strokeDashoffset = '400';
        };

        /* ========================================================
           HOOKS: conectar efectos al ciclo de análisis
        ======================================================== */
        /* Interceptar el botón de escaneo de archivo */
        var scanBtnEl = document.getElementById('scanBtn');
        if (scanBtnEl) {
            var _origClick = scanBtnEl.onclick;
            scanBtnEl.onclick = function (e) {
                if (_origClick) _origClick.call(this, e);
            };
        }

        /* Interceptar el botón de analizar URL */
        var scanUrlBtnEl = document.getElementById('scanUrlBtn');
        if (scanUrlBtnEl) {
            var _origUrlClick = scanUrlBtnEl.onclick;
            scanUrlBtnEl.onclick = function (e) {
                if (_origUrlClick) _origUrlClick.call(this, e);
            };
        }

        /* Observer en ui-result: cuando aparece, disparar efectos */
        var uiResult = document.getElementById('ui-result');
        if (uiResult && typeof MutationObserver !== 'undefined') {
            new MutationObserver(function (mutations) {
                mutations.forEach(function (m) {
                    if (m.attributeName === 'class') {
                        var isVisible = !uiResult.classList.contains('d-none');
                        if (isVisible) {
                            /* Determinar si es IA para glitch */
                            var mainPanel = document.getElementById('mainPanel');
                            var isIA = mainPanel && mainPanel.classList.contains('result-ia');
                            if (typeof window.talosGlitchVerdict === 'function') window.talosGlitchVerdict(isIA);

                            /* Animar firma */
                            if (typeof window.talosAnimateSignature === 'function') window.talosAnimateSignature();
                        } else {
                            /* Reset firma al volver al scanner */
                            if (typeof window.talosResetSignature === 'function') window.talosResetSignature();
                        }
                    }
                });
            }).observe(uiResult, { attributes: true });
        }

    }); // fin waitForAnime

})();
