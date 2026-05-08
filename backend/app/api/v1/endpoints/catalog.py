from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import require_roles
from app.models.catalog import Antibiotic, Microbe
from app.models.user import User, UserRole
from app.schemas.catalog import AntibioticCreate, AntibioticRead, MicrobeCreate, MicrobeRead
from app.services.audit import create_audit_log

router = APIRouter(prefix='/catalog', tags=['catalog'])


@router.get('/microbes', response_model=list[MicrobeRead])
async def list_microbes(db: AsyncSession = Depends(get_db_session)):
    return (await db.scalars(select(Microbe).order_by(Microbe.name.asc()))).all()


@router.post('/microbes', response_model=MicrobeRead)
async def create_microbe(
    payload: MicrobeCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.RESEARCHER, UserRole.LAB_TECHNICIAN)),
):
    microbe = Microbe(**payload.model_dump())
    db.add(microbe)
    await db.flush()
    await create_audit_log(db, 'microbe.created', 'microbe', microbe.id, current_user.id, {'name': microbe.name})
    await db.commit()
    await db.refresh(microbe)
    return microbe


@router.get('/antibiotics', response_model=list[AntibioticRead])
async def list_antibiotics(db: AsyncSession = Depends(get_db_session)):
    return (await db.scalars(select(Antibiotic).order_by(Antibiotic.name.asc()))).all()


@router.post('/antibiotics', response_model=AntibioticRead)
async def create_antibiotic(
    payload: AntibioticCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.RESEARCHER)),
):
    antibiotic = Antibiotic(**payload.model_dump())
    db.add(antibiotic)
    await db.flush()
    await create_audit_log(db, 'antibiotic.created', 'antibiotic', antibiotic.id, current_user.id, {'name': antibiotic.name})
    await db.commit()
    await db.refresh(antibiotic)
    return antibiotic
