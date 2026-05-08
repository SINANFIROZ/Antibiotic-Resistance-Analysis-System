from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog


async def create_audit_log(
    db: AsyncSession,
    action: str,
    entity_type: str,
    entity_id: str | None,
    actor_id: str | None,
    details: dict | None = None,
) -> AuditLog:
    log = AuditLog(
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        actor_id=actor_id,
        details_json=details or {},
    )
    db.add(log)
    await db.flush()
    return log
