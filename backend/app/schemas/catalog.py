from pydantic import BaseModel

from app.schemas.common import TimestampedModel


class MicrobeCreate(BaseModel):
    name: str
    taxonomy_group: str | None = None
    genome_reference: str | None = None
    baseline_resistance_rate: float = 0.5


class AntibioticCreate(BaseModel):
    name: str
    antibiotic_class: str | None = None
    who_awarea_category: str | None = None


class MicrobeRead(TimestampedModel):
    name: str
    taxonomy_group: str | None
    genome_reference: str | None
    baseline_resistance_rate: float


class AntibioticRead(TimestampedModel):
    name: str
    antibiotic_class: str | None
    who_awarea_category: str | None
