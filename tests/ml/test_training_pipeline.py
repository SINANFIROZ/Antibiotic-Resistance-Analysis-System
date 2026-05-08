from pathlib import Path

import pandas as pd

from amr_platform_ml.dataset import load_dataset
from amr_platform_ml.training import train_model_suite


def test_training_pipeline_runs_on_sample_dataset(tmp_path: Path):
    sample = pd.DataFrame(
        [
            {'species': 'e.coli', 'facility_name': 'A', 'mic_value': 2.0, 'target': 1},
            {'species': 'e.coli', 'facility_name': 'B', 'mic_value': 1.0, 'target': 0},
            {'species': 'k.pneumoniae', 'facility_name': 'A', 'mic_value': 4.0, 'target': 1},
            {'species': 'k.pneumoniae', 'facility_name': 'B', 'mic_value': 0.5, 'target': 0},
            {'species': 'p.aeruginosa', 'facility_name': 'A', 'mic_value': 8.0, 'target': 1},
            {'species': 'p.aeruginosa', 'facility_name': 'C', 'mic_value': 0.25, 'target': 0},
        ]
    )
    dataset_path = tmp_path / 'sample.csv'
    sample.to_csv(dataset_path, index=False)

    bundle = load_dataset(dataset_path, target_column='target')
    results = train_model_suite(bundle, tmp_path / 'artifacts')

    assert results
    assert all(Path(result.artifact_path).exists() for result in results)
