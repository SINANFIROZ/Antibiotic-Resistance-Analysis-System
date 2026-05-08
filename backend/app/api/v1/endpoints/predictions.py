from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import require_roles
from app.models.prediction import Prediction
from app.models.user import User, UserRole
from app.schemas.prediction import PredictionRead, PredictionRequest
from app.services.prediction import create_prediction

router = APIRouter(prefix='/predictions', tags=['predictions'])


@router.get('/', response_model=list[PredictionRead])
async def list_predictions(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.ADMIN, UserRole.DOCTOR, UserRole.RESEARCHER, UserRole.LAB_TECHNICIAN)),
):
    return (await db.scalars(select(Prediction).order_by(Prediction.created_at.desc()))).all()


@router.post('/', response_model=PredictionRead)
async def run_prediction(
    payload: PredictionRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.DOCTOR, UserRole.RESEARCHER, UserRole.LAB_TECHNICIAN)),
):
    try:
        return await create_prediction(db, payload, current_user)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
