from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.catalog import Antibiotic, Microbe

settings = get_settings()


def _discover_legacy_antibiotics() -> list[str]:
    antibiotics: set[str] = set()
    for model_dir in settings.legacy_model_dirs:
        if model_dir.exists():
            for path in model_dir.glob('model_*.pkl'):
                antibiotics.add(path.stem.replace('model_', ''))
    return sorted(antibiotics)


def _discover_legacy_microbes() -> list[str]:
    try:
        import joblib
    except ImportError:
        return []

    for model_dir in settings.legacy_model_dirs:
        encoder_path = model_dir / 'species_encoder.pkl'
        if encoder_path.exists():
            encoder = joblib.load(encoder_path)
            return sorted(str(item) for item in getattr(encoder, 'classes_', []))
    return []


async def seed_reference_catalog(db: AsyncSession) -> None:
    if not settings.seed_reference_catalog:
        return

    microbes_in_db = set((await db.scalars(select(Microbe.name))).all())
    antibiotics_in_db = set((await db.scalars(select(Antibiotic.name))).all())

    for microbe_name in _discover_legacy_microbes():
        if microbe_name not in microbes_in_db:
            db.add(Microbe(name=microbe_name, taxonomy_group='clinical isolate', baseline_resistance_rate=0.5))

    for antibiotic_name in _discover_legacy_antibiotics():
        if antibiotic_name not in antibiotics_in_db:
            db.add(Antibiotic(name=antibiotic_name, antibiotic_class='unclassified'))

    await db.commit()
