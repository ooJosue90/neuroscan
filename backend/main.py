from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes_analyze import router as analyze_router
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> Iniciando pre-carga de modelos AI (Lifespan)...")
    
    # 1. Cargar Video Engine (Carga HaarCascades y detector de frames)
    from models.video_detector import get_detector
    get_detector()
    
    # 2. Cargar Image Engine (Carga CLIP y detector SDXL)
    from models.image_detector import get_engine as get_image_engine
    get_image_engine()
    
    # 3. Cargar Audio Pipeline (Carga clasificador HF)
    from models.audio_detector import get_audio_pipeline
    get_audio_pipeline()
    
    print(">>> TODOS LOS MODELOS IA CARGADOS EXITOSAMENTE. Servidor listo.")
    yield
    print(">>> Apagando servidor y liberando recursos...")

app = FastAPI(
    title="NEURO-SCAN PRO",
    description="Detector forense de contenido IA",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)

@app.get("/")
def root():
    return {"message": "NEURO-SCAN PRO API"}