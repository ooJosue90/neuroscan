function switchAuth(mode) {
    clearErrors();
    document.getElementById('login-container').classList.add('d-none');
    document.getElementById('register-container').classList.add('d-none');
    document.getElementById('reset-container').classList.add('d-none');

    if (mode === 'login') document.getElementById('login-container').classList.remove('d-none');
    else if (mode === 'register') document.getElementById('register-container').classList.remove('d-none');
    else if (mode === 'reset') document.getElementById('reset-container').classList.remove('d-none');
}

function showFieldError(inputId, msg) {
    const errorDiv = document.getElementById('error-' + inputId);
    if (errorDiv) {
        errorDiv.textContent = msg;
        errorDiv.classList.add('show');
    }
}

function clearErrors() {
    document.querySelectorAll('.error-msg-talos').forEach(el => {
        el.textContent = '';
        el.classList.remove('show');
    });
}

// --- Nueva función para mostrar/ocultar contraseña ---
function togglePassword(inputId, btn) {
    const input = document.getElementById(inputId);
    const icon = btn.querySelector('i');
    if (input.type === 'password') {
        input.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    } else {
        input.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    }
}

async function handleLogin() {
    clearErrors();
    const user = document.getElementById('userLogin').value.trim();
    const pass = document.getElementById('passLogin').value.trim();
    
    let hasError = false;
    if (!user) { showFieldError('userLogin', "Identificación requerida"); hasError = true; }
    if (!pass) { showFieldError('passLogin', "Código de seguridad requerido"); hasError = true; }
    if (hasError) return;

    try {
        const res = await fetch('http://127.0.0.1:8000/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, password: pass })
        });
        const data = await res.json();
        if (res.ok) {
            localStorage.setItem('talos_session', data.username);
            window.location.href = 'index.html';
        } else {
            alert("ACCESO DENEGADO: " + (data.detail || "Error fatal."));
        }
    } catch (e) { alert("ERROR: Sin respuesta del núcleo."); }
}

async function handleRegister() {
    clearErrors();
    const user = document.getElementById('userReg').value.trim();
    const pass = document.getElementById('passReg').value.trim();
    const passConfirm = document.getElementById('passConfirmReg').value.trim();
    const email = document.getElementById('emailReg').value.trim();
    const name = document.getElementById('nameReg').value.trim();
    const terms = document.getElementById('terms').checked;

    let hasError = false;
    if (!name) { showFieldError('nameReg', "Nombre obligatorio"); hasError = true; }
    if (!email) { showFieldError('emailReg', "Correo obligatorio"); hasError = true; }
    if (!user) { showFieldError('userReg', "Usuario obligatorio"); hasError = true; }
    if (!pass) { showFieldError('passReg', "Contraseña obligatoria"); hasError = true; }
    if (!passConfirm) { showFieldError('passConfirmReg', "Verificación obligatoria"); hasError = true; }
    
    if (!hasError && pass !== passConfirm) {
        showFieldError('passConfirmReg', "Los códigos no coinciden");
        hasError = true;
    }

    if (!terms) {
        showFieldError('terms', "Debe aceptar los protocolos");
        hasError = true;
    }
    
    if (hasError) return;


    try {
        const res = await fetch('http://127.0.0.1:8000/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                username: user, password: pass,
                email: email, full_name: name
            })
        });
        const data = await res.json();
        if (res.ok) {
            alert("EXPEDIENTE CREADO. Ya puede acceder.");
            switchAuth('login');
        } else {
            alert("ERROR: " + (data.detail || "Registro fallido."));
        }
    } catch (e) { alert("ERROR: Fallo crítico de conexión."); }
}

async function handleReset() {
    clearErrors();
    const email = document.getElementById('emailReset').value.trim();
    if (!email) {
        showFieldError('emailReset', "Correo requerido para protocolo");
        return;
    }

    alert("PROTOCOLO INICIADO: Si el correo existe en el núcleo, recibirá las instrucciones en breve.");
    switchAuth('login');
}
