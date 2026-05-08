from datetime import date

from pydantic import BaseModel, Field

from app.schemas.common import TimestampedModel


class PatientCreate(BaseModel):
    external_id: str = Field(min_length=2, max_length=100)
    first_name: str
    last_name: str
    date_of_birth: date
    sex: str
    facility_name: str


class PatientRead(TimestampedModel):
    external_id: str
    first_name: str
    last_name: str
    date_of_birth: date
    sex: str
    facility_name: str
    created_by_id: str
