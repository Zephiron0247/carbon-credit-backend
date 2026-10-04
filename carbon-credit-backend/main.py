import os
import threading
import time

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import engine, Base
from routes.projects import router as projects_router
from routes.credits import router as credits_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Carbon Credit Verification API",
    description="Satellite-ML-Blockchain MRV platform for carbon credit verification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "https://carbon-admin-dashboard.onrender.com",
        "https://carbon-company-dashboard.onrender.com",
        "https://carbon-buyer-dashboard.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router)
app.include_router(credits_router)


@app.get("/")
def root():
    return {"message": "Carbon Credit Verification API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ping")
def ping():
    return {"pong": True}


def _keep_alive():
    url = os.getenv("RENDER_EXTERNAL_URL")
    if not url:
        return
    while True:
        time.sleep(600)
        try:
            requests.get(f"{url}/ping", timeout=10)
        except Exception:
            pass


@app.on_event("startup")
def _start_keep_alive():
    threading.Thread(target=_keep_alive, daemon=True).start()