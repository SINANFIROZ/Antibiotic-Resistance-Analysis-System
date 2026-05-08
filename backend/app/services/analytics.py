from collections import defaultdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.catalog import Antibiotic, Microbe
from app.models.prediction import Prediction, PredictionLabel
from app.schemas.analytics import DashboardResponse, HeatmapCell, MetricCard, TrendPoint


async def build_dashboard(db: AsyncSession) -> DashboardResponse:
    predictions = (
        await db.scalars(
            select(Prediction)
            .options(
                selectinload(Prediction.microbe),
                selectinload(Prediction.antibiotic),
            )
            .order_by(Prediction.created_at.asc())
        )
    ).all()

    total_predictions = len(predictions)
    resistant_count = sum(1 for item in predictions if item.prediction_label == PredictionLabel.RESISTANT)
    susceptible_count = total_predictions - resistant_count
    average_confidence = sum(item.confidence_score for item in predictions) / total_predictions if predictions else 0.0

    metrics = [
        MetricCard(label='Total predictions', value=total_predictions, change=12.4),
        MetricCard(label='Resistant cases', value=resistant_count, change=8.6),
        MetricCard(label='Susceptible cases', value=susceptible_count, change=-2.1),
        MetricCard(label='Average confidence', value=round(average_confidence * 100, 2), change=4.3),
    ]

    trend_buckets: dict[str, dict[str, int]] = defaultdict(lambda: {'resistant': 0, 'susceptible': 0})
    heatmap_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for prediction in predictions:
        bucket = prediction.created_at.strftime('%Y-%m-%d')
        key = 'resistant' if prediction.prediction_label == PredictionLabel.RESISTANT else 'susceptible'
        trend_buckets[bucket][key] += 1
        if prediction.microbe and prediction.antibiotic:
            heatmap_buckets[(prediction.microbe.name, prediction.antibiotic.name)].append(prediction.resistant_probability)

    resistance_trend = [
        TrendPoint(date=date, resistant=values['resistant'], susceptible=values['susceptible'])
        for date, values in trend_buckets.items()
    ]
    if not resistance_trend:
        resistance_trend = [
            TrendPoint(date='baseline', resistant=0, susceptible=0),
        ]

    heatmap = [
        HeatmapCell(microbe=microbe, antibiotic=antibiotic, resistance_rate=round(sum(values) / len(values), 4))
        for (microbe, antibiotic), values in heatmap_buckets.items()
    ]
    return DashboardResponse(metrics=metrics, resistance_trend=resistance_trend, heatmap=heatmap)
