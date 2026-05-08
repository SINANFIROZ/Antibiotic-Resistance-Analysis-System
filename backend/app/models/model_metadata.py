from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ModelMetadata(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'model_metadata'

    model_name: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[str] = mapped_column(String(100), index=True)
    algorithm: Mapped[str] = mapped_column(String(120))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
