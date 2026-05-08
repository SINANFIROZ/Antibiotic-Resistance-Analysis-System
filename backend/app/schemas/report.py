from datetime import datetime

from pydantic import BaseModel

from app.models.report import ReportStatus


class ReportRead(BaseModel):
    id: str
    created_at: datetime
    prediction_id: str
    created_by_id: str
    file_path: str
    status: ReportStatus
    emailed_to: str | None
