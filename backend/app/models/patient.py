from datetime import date

from sqlalchemy import Date, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Patient(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'patients'

    external_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    first_name: Mapped[str] = mapped_column(String(120))
    last_name: Mapped[str] = mapped_column(String(120))
    date_of_birth: Mapped[date] = mapped_column(Date)
    sex: Mapped[str] = mapped_column(String(20))
    facility_name: Mapped[str] = mapped_column(String(255), default='Unknown Facility')
    created_by_id: Mapped[str] = mapped_column(ForeignKey('users.id'))

    created_by = relationship('User', back_populates='patients')
    predictions = relationship('Prediction', back_populates='patient')
