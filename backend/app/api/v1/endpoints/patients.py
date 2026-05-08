from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import get_current_user, require_roles
from app.models.patient import Patient
from app.models.user import User, UserRole
from app.schemas.patient import PatientCreate, PatientRead
from app.services.audit import create_audit_log

router = APIRouter(prefix='/patients', tags=['patients'])


@router.get('/', response_model=list[PatientRead])
async def list_patients(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.ADMIN, UserRole.DOCTOR, UserRole.LAB_TECHNICIAN)),
):
    return (await db.scalars(select(Patient).order_by(Patient.created_at.desc()))).all()


@router.post('/', response_model=PatientRead)
async def create_patient(
    payload: PatientCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.DOCTOR, UserRole.LAB_TECHNICIAN)),
):
    patient = Patient(**payload.model_dump(), created_by_id=current_user.id)
    db.add(patient)
    await db.flush()
    await create_audit_log(db, 'patient.created', 'patient', patient.id, current_user.id, {'external_id': patient.external_id})
    await db.commit()
    await db.refresh(patient)
    return patient
