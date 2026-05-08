from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ORMBaseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    message: str


class TimestampedModel(ORMBaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
