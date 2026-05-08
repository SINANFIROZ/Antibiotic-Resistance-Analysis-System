from datetime import datetime

from pydantic import BaseModel


class DatasetRead(BaseModel):
    id: str
    created_at: datetime
    filename: str
    source: str
    schema_version: str
    row_count: int
    status: str
    notes: str | None
    uploaded_by_id: str
