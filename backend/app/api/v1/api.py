from fastapi import APIRouter

from app.api.v1.endpoints import analytics, audit, auth, catalog, datasets, patients, predictions, reports, users

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(patients.router)
api_router.include_router(catalog.router)
api_router.include_router(predictions.router)
api_router.include_router(analytics.router)
api_router.include_router(reports.router)
api_router.include_router(datasets.router)
api_router.include_router(audit.router)
