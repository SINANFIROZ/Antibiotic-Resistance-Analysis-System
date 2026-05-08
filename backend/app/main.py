from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.api import api_router
from app.core.config import get_settings
from app.core.database import AsyncSessionLocal, Base, engine
from app.core.logging import configure_logging
from app.services.bootstrap import bootstrap_application

settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=lambda request: request.client.host if request.client else 'unknown')


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_schema:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:
        await bootstrap_application(session)
    yield


app = FastAPI(
    title=settings.project_name,
    version='1.0.0',
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    docs_url=f"{settings.api_v1_prefix}/docs",
    redoc_url=f"{settings.api_v1_prefix}/redoc",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.middleware('http')
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception('Unhandled exception', exc_info=exc)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error'})


@app.get('/health/live')
async def liveness() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/health/ready')
async def readiness() -> dict[str, str]:
    return {'status': 'ready'}


app.include_router(api_router, prefix=settings.api_v1_prefix)
