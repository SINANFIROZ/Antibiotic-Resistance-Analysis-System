# Setup Guide

## Backend
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
PYTHONPATH=backend uvicorn app.main:app --reload --app-dir backend
```

## Frontend
```bash
cd frontend
npm install
npm run dev
```

## Full stack with Docker
```bash
docker compose up --build
```
