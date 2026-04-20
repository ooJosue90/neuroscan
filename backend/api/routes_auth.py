from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models.auth import get_db, create_user, authenticate_user, User

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterSchema(BaseModel):
    username: str
    password: str
    email: str
    full_name: str

@router.post("/register")
def register(auth: RegisterSchema, db: Session = Depends(get_db)):
    # Verificar si usuario o email ya existen
    if db.query(User).filter(User.username == auth.username).first():
        raise HTTPException(status_code=400, detail="El ID de operador ya está en uso.")
    if db.query(User).filter(User.email == auth.email).first():
        raise HTTPException(status_code=400, detail="El correo electrónico ya está registrado.")
        
    user = create_user(db, auth.username, auth.password, auth.email, auth.full_name)
    print(f"👤 NUEVO USUARIO CREADO: El operador '{user.username}' ({user.full_name}) ha sido registrado en el sistema.")
    return {"status": "success", "username": user.username}

class AuthSchema(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(auth: AuthSchema, db: Session = Depends(get_db)):
    user = authenticate_user(db, auth.username, auth.password)
    if not user:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    print(f"🌐 LOGIN DETECTADO: El usuario '{user.username}' ha iniciado sesión.")
    return {"status": "success", "username": user.username, "token": "talos_session_token_example"} # Simplificado para el demo

class LogoutSchema(BaseModel):
    username: str

@router.post("/logout")
def logout(data: LogoutSchema):
    print(f"🛑 LOGOUT DETECTADO: El usuario '{data.username}' ha cerrado sesión en el sistema TALOS.")
    return {"status": "success", "message": "Sesión cerrada en el servidor"}
