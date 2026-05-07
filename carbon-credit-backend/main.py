from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routes.projects import router as projects_router
from routes.credits import router as credits_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title       = "Carbon Credit Verification API",
    description = "Satellite-ML-Blockchain MRV platform for carbon credit verification",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(projects_router)
app.include_router(credits_router)

@app.get("/")
def root():
    return {"message": "Carbon Credit Verification API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}