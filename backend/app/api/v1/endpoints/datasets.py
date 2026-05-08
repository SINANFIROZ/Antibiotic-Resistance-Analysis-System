from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import require_roles
from app.models.dataset import UploadedDataset
from app.models.user import User, UserRole
from app.schemas.dataset import DatasetRead
from app.services.datasets import store_dataset

router = APIRouter(prefix='/datasets', tags=['datasets'])


@router.get('/', response_model=list[DatasetRead])
async def list_datasets(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.ADMIN, UserRole.RESEARCHER, UserRole.LAB_TECHNICIAN)),
):
    return (await db.scalars(select(UploadedDataset).order_by(UploadedDataset.created_at.desc()))).all()


@router.post('/upload', response_model=DatasetRead)
async def upload_dataset(
    notes: str | None = Form(default=None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.RESEARCHER, UserRole.LAB_TECHNICIAN)),
):
    return await store_dataset(db, file, current_user, notes)
