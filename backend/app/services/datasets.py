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
    if not file.filename:
        raise ValueError('Uploaded file must have a filename')
    safe_filename = Path(file.filename).name
    if not safe_filename:
        raise ValueError('Uploaded file must have a valid filename')

    content = await file.read()

    row_count = 0
    if safe_filename.endswith('.csv'):
        try:
            decoded_content = content.decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValueError('Uploaded CSV files must be UTF-8 encoded. Please convert your file to UTF-8 encoding and try again.') from error
        csv_rows = list(csv.reader(decoded_content.splitlines()))
        row_count = max(len(csv_rows) - 1, 0)

    settings.dataset_storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.dataset_storage_dir / safe_filename
    file_path.write_bytes(content)

    dataset = UploadedDataset(
        filename=safe_filename,
        source='manual_upload',
        schema_version='v1',
        row_count=row_count,
        status='uploaded',
        notes=notes,
        uploaded_by_id=actor.id,
    )
    db.add(dataset)
    await db.flush()
    await create_audit_log(db, 'dataset.uploaded', 'dataset', dataset.id, actor.id, {'filename': safe_filename, 'row_count': row_count})
    await db.commit()
    await db.refresh(dataset)
    return dataset
