from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
import os

# Base de datos SQLite local dentro de la carpeta data
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    password_hash = Column(String)

# Crear tablas
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(db, username, password, email=None, full_name=None):
    hashed_pass = hash_password(password)
    db_user = User(username=username, password_hash=hashed_pass, email=email, full_name=full_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db, username, password):
    user = db.query(User).filter(User.username == username).first()
    if user and user.password_hash == hash_password(password):
        return user
    return None

# Crear usuario por defecto si no hay ninguno
def seed_default_user():
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            print(">>> Base de datos vacía. Creando operador maestro (admin)...")
            create_user(db, "admin", "talos2026", "admin@talos.ai", "Administrador Maestro")
    finally:
        db.close()

seed_default_user()
