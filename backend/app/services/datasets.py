import csv
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.dataset import UploadedDataset
from app.models.user import User
from app.services.audit import create_audit_log

settings = get_settings()


async def store_dataset(db: AsyncSession, file: UploadFile, actor: User, notes: str | None = None) -> UploadedDataset:
    settings.dataset_storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.dataset_storage_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    row_count = 0
    if file.filename.endswith('.csv'):
        row_count = max(sum(1 for _ in csv.reader(content.decode('utf-8', errors='ignore').splitlines())) - 1, 0)

    dataset = UploadedDataset(
        filename=file.filename,
        source='manual_upload',
        schema_version='v1',
        row_count=row_count,
        status='uploaded',
        notes=notes,
        uploaded_by_id=actor.id,
    )
    db.add(dataset)
    await db.flush()
    await create_audit_log(db, 'dataset.uploaded', 'dataset', dataset.id, actor.id, {'filename': file.filename, 'row_count': row_count})
    await db.commit()
    await db.refresh(dataset)
    return dataset
