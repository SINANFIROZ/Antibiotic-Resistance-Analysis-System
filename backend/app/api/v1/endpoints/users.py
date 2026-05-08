from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import get_current_user, require_roles
from app.models.user import User, UserRole
from app.schemas.user import UserRead

router = APIRouter(prefix='/users', tags=['users'])


@router.get('/', response_model=list[UserRead])
async def list_users(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.ADMIN)),
):
    return (await db.scalars(select(User).order_by(User.created_at.desc()))).all()


@router.get('/me', response_model=UserRead)
async def current_user_profile(current_user: User = Depends(get_current_user)):
    return current_user
