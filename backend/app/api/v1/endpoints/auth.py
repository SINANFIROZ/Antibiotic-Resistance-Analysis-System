from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.security import create_access_token, create_refresh_token, decode_token
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.schemas.auth import AuthenticatedUser, TokenRefreshRequest, TokenResponse, UserLogin, UserRegister
from app.schemas.user import UserRead
from app.services.audit import create_audit_log
from app.services.auth import authenticate_user, register_user

router = APIRouter(prefix='/auth', tags=['auth'])
limiter = Limiter(key_func=lambda request: request.client.host if request.client else 'unknown')


@router.post('/register', response_model=UserRead)
@limiter.limit('10/minute')
async def register(
    request: Request,
    payload: UserRegister,
    db: AsyncSession = Depends(get_db_session),
):
    return await register_user(db, payload)


@router.post('/login', response_model=TokenResponse)
@limiter.limit('10/minute')
async def login(
    request: Request,
    payload: UserLogin,
    db: AsyncSession = Depends(get_db_session),
):
    return await authenticate_user(db, payload)


@router.post('/refresh', response_model=TokenResponse)
@limiter.limit('20/minute')
async def refresh_token(
    request: Request,
    payload: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    try:
        token_payload = decode_token(payload.refresh_token)
    except ValueError as error:
        raise HTTPException(status_code=401, detail='Invalid refresh token') from error
    if token_payload.get('type') != 'refresh':
        raise HTTPException(status_code=401, detail='Invalid refresh token')
    user = await db.scalar(select(User).where(User.id == token_payload['sub']))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail='Inactive or missing user')
    await create_audit_log(db, 'token.refreshed', 'user', user.id, user.id, {'source': 'refresh_token'})
    await db.commit()
    return TokenResponse(
        access_token=create_access_token(user.id, user.role.value),
        refresh_token=create_refresh_token(token_payload['sub']),
    )


@router.get('/me', response_model=AuthenticatedUser)
async def read_me(current_user=Depends(get_current_user)):
    return current_user
