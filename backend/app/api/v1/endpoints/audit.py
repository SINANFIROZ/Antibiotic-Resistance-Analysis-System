from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.dependencies.auth import require_roles
from app.models.audit_log import AuditLog
from app.models.user import User, UserRole

router = APIRouter(prefix='/audit', tags=['audit'])


@router.get('/')
async def list_audit_logs(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.ADMIN)),
):
    logs = (await db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()))).all()
    return [
        {
            'id': log.id,
            'created_at': log.created_at,
            'action': log.action,
            'entity_type': log.entity_type,
            'entity_id': log.entity_id,
            'actor_id': log.actor_id,
            'details_json': log.details_json,
        }
        for log in logs
    ]
