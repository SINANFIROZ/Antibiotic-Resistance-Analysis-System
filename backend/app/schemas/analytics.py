from pydantic import BaseModel


class MetricCard(BaseModel):
    label: str
    value: float
    change: float


class TrendPoint(BaseModel):
    date: str
    resistant: int
    susceptible: int


class HeatmapCell(BaseModel):
    microbe: str
    antibiotic: str
    resistance_rate: float


class DashboardResponse(BaseModel):
    metrics: list[MetricCard]
    resistance_trend: list[TrendPoint]
    heatmap: list[HeatmapCell]
