from pydantic import EmailStr

from app.models.user import UserRole
from app.schemas.common import TimestampedModel


class UserRead(TimestampedModel):
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
