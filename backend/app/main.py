import sys
from pathlib import Path

# Ensure root edge-ai-vision directory is in sys.path
ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.database.session import init_db
from app.api.v1 import (
    auth, cameras, models_router, pipelines, events, system, devices, ws, users_audit, datasets, experiments, capture_manager, annotations, gate
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("edge-vision")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Modular Edge-AI Vision Monitoring Platform...")
    # Initialize DB tables and seed data
    await init_db()
    logger.info("Database initialized successfully.")
    yield
    logger.info("Shutting down Edge-AI Platform...")

app = FastAPI(
    title=settings.APP_NAME,
    description="Modular Edge-AI Vision Monitoring Platform for Hikvision and other edge IP cameras",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for datasets & snapshots
datasets_dir = Path(settings.STORAGE_DIR) / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")
app.mount("/storage", StaticFiles(directory=str(settings.STORAGE_DIR)), name="storage")

# Register API v1 Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(cameras.router, prefix=f"{settings.API_V1_STR}/cameras", tags=["Cameras"])
app.include_router(capture_manager.router, prefix=f"{settings.API_V1_STR}", tags=["Dataset Capture Manager"])
app.include_router(annotations.router, prefix=f"{settings.API_V1_STR}", tags=["Annotation Studio & Review"])
app.include_router(models_router.router, prefix=f"{settings.API_V1_STR}/models", tags=["Models"])
app.include_router(pipelines.router, prefix=f"{settings.API_V1_STR}/pipelines", tags=["Pipelines"])
app.include_router(events.router, prefix=f"{settings.API_V1_STR}/events", tags=["Recognition Events"])
app.include_router(gate.router, prefix=f"{settings.API_V1_STR}/gate", tags=["UNSGate Operations"])
app.include_router(datasets.router, prefix=f"{settings.API_V1_STR}/datasets", tags=["Datasets"])
app.include_router(experiments.router, prefix=f"{settings.API_V1_STR}/experiments", tags=["Experiments"])
app.include_router(devices.router, prefix=f"{settings.API_V1_STR}", tags=["Devices & Gate"])
app.include_router(system.router, prefix=f"{settings.API_V1_STR}", tags=["System & Health"])
app.include_router(users_audit.router, prefix=f"{settings.API_V1_STR}", tags=["Users & Audit"])
app.include_router(ws.router, prefix=f"{settings.API_V1_STR}", tags=["WebSockets"])

@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
