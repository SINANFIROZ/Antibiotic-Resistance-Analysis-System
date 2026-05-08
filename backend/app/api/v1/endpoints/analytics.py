from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import require_roles
from app.models.user import User, UserRole
from app.schemas.analytics import DashboardResponse
from app.services.analytics import build_dashboard

router = APIRouter(prefix='/analytics', tags=['analytics'])


@router.get('/dashboard', response_model=DashboardResponse)
async def dashboard(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.ADMIN, UserRole.RESEARCHER, UserRole.DOCTOR)),
):
    return await build_dashboard(db)
