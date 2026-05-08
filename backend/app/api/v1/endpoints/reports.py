from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import require_roles
from app.models.report import Report
from app.models.user import User, UserRole
from app.schemas.report import ReportRead
from app.services.reporting import generate_report

router = APIRouter(prefix='/reports', tags=['reports'])


@router.get('/', response_model=list[ReportRead])
async def list_reports(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.ADMIN, UserRole.DOCTOR, UserRole.RESEARCHER)),
):
    return (await db.scalars(select(Report).order_by(Report.created_at.desc()))).all()


@router.post('/{prediction_id}', response_model=ReportRead)
async def create_report(
    prediction_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.DOCTOR, UserRole.RESEARCHER)),
):
    try:
        return await generate_report(db, prediction_id, current_user)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
