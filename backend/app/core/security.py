from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_token(subject: str, expires_delta: timedelta, token_type: str, extra_claims: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        'sub': subject,
        'type': token_type,
        'exp': datetime.now(UTC) + expires_delta,
        'iat': datetime.now(UTC),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str, role: str) -> str:
    return create_token(
        subject=subject,
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
        token_type='access',
        extra_claims={'role': role},
    )


def create_refresh_token(subject: str) -> str:
    return create_token(
        subject=subject,
        expires_delta=timedelta(minutes=settings.refresh_token_expire_minutes),
        token_type='refresh',
    )


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as error:
        raise ValueError('Invalid authentication token') from error
