import enum

from sqlalchemy import Boolean, Enum, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class UserRole(str, enum.Enum):
    ADMIN = 'admin'
    RESEARCHER = 'researcher'
    DOCTOR = 'doctor'
    LAB_TECHNICIAN = 'lab_technician'


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'users'

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.DOCTOR, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    patients = relationship('Patient', back_populates='created_by')
    predictions = relationship('Prediction', back_populates='requested_by')
    reports = relationship('Report', back_populates='created_by')
    datasets = relationship('UploadedDataset', back_populates='uploaded_by')
