from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import hash_password
from app.models.user import User, UserRole
from app.services.audit import create_audit_log
from app.services.catalog import seed_reference_catalog

settings = get_settings()


async def bootstrap_application(db: AsyncSession) -> None:
    await seed_reference_catalog(db)

    existing_admin = await db.scalar(select(User).where(User.email == settings.initial_admin_email))
    if existing_admin:
        return

    admin = User(
        email=settings.initial_admin_email,
        full_name='Platform Administrator',
        hashed_password=hash_password(settings.initial_admin_password),
        role=UserRole.ADMIN,
    )
    db.add(admin)
    await db.flush()
    await create_audit_log(db, 'platform.admin_seeded', 'user', admin.id, admin.id, {'email': admin.email})
    await db.commit()
