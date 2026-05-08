from sqlalchemy import DateTime, Float, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class AnalyticsSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'analytics_snapshots'

    metric_name: Mapped[str] = mapped_column(String(120), index=True)
    time_bucket: Mapped[str] = mapped_column(String(50))
    value: Mapped[float] = mapped_column(Float)
    dimensions_json: Mapped[dict] = mapped_column(JSON, default=dict)
