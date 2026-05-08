from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from amr_platform_ml.dataset import DatasetBundle


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    artifact_path: str


def _optional_model(module_name: str, class_name: str, **kwargs):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)(**kwargs)
    except (ImportError, AttributeError):
        return None


def _build_estimators(random_state: int = 42) -> dict[str, object]:
    estimators: dict[str, object] = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=random_state),
        'logistic_regression': LogisticRegression(max_iter=1000),
    }
    xgb = _optional_model('xgboost', 'XGBClassifier', n_estimators=200, random_state=random_state, eval_metric='logloss')
    if xgb is not None:
        estimators['xgboost'] = xgb
    lgbm = _optional_model('lightgbm', 'LGBMClassifier', n_estimators=250, random_state=random_state)
    if lgbm is not None:
        estimators['lightgbm'] = lgbm
    catboost = _optional_model('catboost', 'CatBoostClassifier', random_state=random_state, verbose=0)
    if catboost is not None:
        estimators['catboost'] = catboost
    if len(estimators) >= 2:
        estimators['ensemble'] = VotingClassifier(
            estimators=[(name, estimator) for name, estimator in estimators.items() if name != 'ensemble'],
            voting='soft',
        )
    return estimators


def _calculate_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auprc': average_precision_score(y_true, y_prob),
    }
    if len(set(y_true)) > 1:
        metrics['auroc'] = roc_auc_score(y_true, y_prob)
    return {key: float(round(value, 4)) for key, value in metrics.items()}


def train_model_suite(bundle: DatasetBundle, output_dir: str | Path, random_state: int = 42) -> list[ModelResult]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    X = bundle.dataframe[bundle.feature_columns]
    y = bundle.dataframe[bundle.target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    results: list[ModelResult] = []
    for name, estimator in _build_estimators(random_state=random_state).items():
        pipeline = Pipeline([
            ('preprocess', bundle.preprocessing_pipeline),
            ('model', estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = _calculate_metrics(y_test, predictions, probabilities)
        cv_scores = cross_val_score(pipeline, X, y, cv=min(3, len(np.unique(y))), scoring='f1')
        metrics['cv_f1_mean'] = float(round(np.mean(cv_scores), 4))
        artifact = output_path / f'{name}.joblib'
        joblib.dump(pipeline, artifact)
        results.append(ModelResult(name=name, metrics=metrics, artifact_path=str(artifact)))
    return results
