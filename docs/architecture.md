# AMR Intelligence Platform Architecture

## Core domains
- **frontend/**: Next.js TypeScript SaaS UI for clinical and research workflows.
- **backend/**: FastAPI API-first application with JWT auth, RBAC, dataset management, analytics, reporting, and prediction endpoints.
- **ml/**: Reproducible training/evaluation pipeline with optional advanced model backends and explainability hooks.
- **analytics/**: Surveillance-oriented derived datasets and research jobs.
- **infrastructure/**: Docker, reverse proxy, and deployment runtime configuration.

## Production qualities
- API versioning under `/api/v1`
- PostgreSQL-compatible async data layer
- Report and dataset persistence in mounted storage
- Security headers, CORS, RBAC, audit logs, and rate limiting
- Dockerized local environment and CI pipeline
