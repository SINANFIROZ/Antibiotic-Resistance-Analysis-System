from __future__ import annotations

from typing import Any

import httpx


class MlflowTrackingClient:
    def __init__(self, tracking_uri: str, experiment_id: str):
        self.tracking_uri = tracking_uri.rstrip('/')
        self.experiment_id = experiment_id

    async def log_run(self, run_name: str, params: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f'{self.tracking_uri}/api/2.0/mlflow/runs/create',
                json={'experiment_id': self.experiment_id, 'run_name': run_name},
            )
            response.raise_for_status()
            run = response.json()['run']
            run_id = run['info']['run_id']
            await client.post(
                f'{self.tracking_uri}/api/2.0/mlflow/runs/log-batch',
                json={
                    'run_id': run_id,
                    'metrics': [{'key': key, 'value': value, 'timestamp': 0, 'step': 0} for key, value in metrics.items()],
                    'params': [{'key': key, 'value': str(value)} for key, value in params.items()],
                },
            )
            return run
