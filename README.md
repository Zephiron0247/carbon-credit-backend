# Pranaq — Backend MRV Engine

FastAPI + ML + Remote Sensing backend powering the Pranaq carbon credit MRV platform.

---

# Features

- Sentinel-2 NDVI/EVI/NDWI analysis
- Multi-year ecological monitoring
- ML confidence scoring
- Fraud detection workflows
- Verification pipelines
- Blockchain integration
- Admin governance support

---

# Stack

- FastAPI
- SQLAlchemy
- Python
- Google Earth Engine
- Web3.py

---

# Core Modules

| Module | Purpose |
|---|---|
| pipeline.py | End-to-end verification pipeline |
| ndvi_pipeline.py | Remote sensing analysis |
| ml_scoring.py | ML confidence scoring |
| routes/projects.py | Verification endpoints |

---

# Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
