# AMR Intelligence Platform

AMR Intelligence Platform is a production-oriented rebuild of the original Antibiotic Resistance Analysis System into a modular healthcare SaaS foundation.

## What changed
- **Frontend**: Next.js + TypeScript + Tailwind dashboard foundation in `frontend/`
- **Backend**: FastAPI API-first service in `backend/` with JWT auth, RBAC, async SQLAlchemy, dataset upload, prediction, reporting, analytics, and audit logging
- **ML**: Reproducible training package in `ml/` with extensible multi-model training and optional explainability integrations
- **Infrastructure**: Docker Compose, Nginx reverse proxy, and GitHub Actions CI
- **Documentation**: Architecture and setup docs under `docs/`

## Product capabilities in this implementation
- Role-aware clinical and research API foundation
- Seeded catalog from legacy AMR artifacts in `Models/`
- Legacy model-backed prediction endpoint with heuristic fallback
- Dashboard analytics API with resistance trend and heatmap aggregates
- PDF report generation and dataset ingestion with audit logs
- PostgreSQL-ready schema with Alembic baseline migration

## Repository layout
- `frontend/` — Next.js SaaS UI
- `backend/` — FastAPI services and Alembic migrations
- `ml/` — training/evaluation pipeline
- `analytics/` — surveillance workflows
- `infrastructure/` — Nginx and deployment assets
- `docs/` — architecture and setup documentation
- `tests/` — backend and ML tests

## Quick start
1. Copy `.env.example` to `.env` and adjust secrets.
2. Start locally with Docker:
   ```bash
   docker compose up --build
   ```
3. Or run services separately using `docs/setup.md`.

## API highlights
- `/health/live`
- `/health/ready`
- `/api/v1/auth/*`
- `/api/v1/catalog/*`
- `/api/v1/patients`
- `/api/v1/predictions`
- `/api/v1/analytics/dashboard`
- `/api/v1/reports/{prediction_id}`
- `/api/v1/datasets/upload`

## Default bootstrap admin
- Email: `admin@amr.local`
- Password: `ChangeMe123!`

Change these immediately via environment variables outside local development.
