"""Save, load, and run exported Persuade ESC sklearn classifiers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Literal

import joblib
import numpy as np

SelectionStrategy = Literal["vote", "average_proba", "any_positive", "all_positive"]


def _as_2d(features: np.ndarray) -> np.ndarray:
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def save_sklearn_model_suite(
    models_full: dict[str, Any],
    feature_names: list[str],
    export_path: str | Path,
    cv_summary: list[dict[str, Any]] | None = None,
    models_cv_folds: dict[str, list[dict[str, Any]]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save all trained models plus optional CV fold models in one bundle."""
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "version": 1,
        "feature_names": list(feature_names),
        "class_labels": {"Negative": False, "Positive": True},
        "model_names": list(models_full.keys()),
        "models_full": models_full,
        "models_cv_folds": models_cv_folds or {},
        "cv_summary": cv_summary or [],
        "metadata": metadata or {},
    }
    joblib.dump(bundle, export_path)
    return export_path


def load_sklearn_model_suite(export_path: str | Path) -> dict[str, Any]:
    return joblib.load(export_path)


def list_models(bundle: dict[str, Any]) -> list[str]:
    return list(bundle.get("model_names") or bundle.get("models_full", {}).keys())


def get_cv_summary(bundle: dict[str, Any], model_name: str | None = None) -> list[dict[str, Any]]:
    summary = bundle.get("cv_summary", [])
    if model_name is None:
        return summary
    return [row for row in summary if row.get("Model") == model_name]


def get_model(
    bundle: dict[str, Any],
    model_name: str,
    *,
    source: Literal["full", "cv_folds"] = "full",
    fold: int | None = None,
) -> Any:
    """Return one fitted pipeline.

    source='full' -> model retrained on all data (recommended for deployment)
    source='cv_folds' -> one CV fold model; fold is 1-based
    """
    if source == "full":
        try:
            return bundle["models_full"][model_name]
        except KeyError as exc:
            raise KeyError(f"Unknown model '{model_name}'. Available: {list_models(bundle)}") from exc

    fold_models = bundle.get("models_cv_folds", {}).get(model_name, [])
    if not fold_models:
        raise KeyError(f"No CV fold models stored for '{model_name}'.")

    if fold is None:
        raise ValueError("fold is required when source='cv_folds'.")

    for entry in fold_models:
        if entry["fold"] == fold:
            return entry["pipeline"]

    available = [entry["fold"] for entry in fold_models]
    raise KeyError(f"Fold {fold} not found for '{model_name}'. Available folds: {available}")


def _predict_one_pipeline(pipeline, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    X = _as_2d(features)
    preds = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X)[:, 1]
    else:
        probas = None
    return preds, probas


def predict_with_models(
    bundle: dict[str, Any],
    model_names: Iterable[str],
    features: np.ndarray,
    *,
    source: Literal["full", "cv_folds"] = "full",
    folds: dict[str, int] | None = None,
    strategy: SelectionStrategy = "vote",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict using one or more exported models."""
    names = list(model_names)
    if not names:
        raise ValueError("model_names must contain at least one model.")

    all_preds = []
    all_probas = []

    for name in names:
        if source == "full":
            pipeline = get_model(bundle, name, source="full")
            pred, proba = _predict_one_pipeline(pipeline, features)
            all_preds.append(pred)
            if proba is not None:
                all_probas.append(proba)
            continue

        fold = (folds or {}).get(name)
        if fold is None:
            raise ValueError(f"Missing fold for '{name}' when source='cv_folds'.")
        pipeline = get_model(bundle, name, source="cv_folds", fold=fold)
        pred, proba = _predict_one_pipeline(pipeline, features)
        all_preds.append(pred)
        if proba is not None:
            all_probas.append(proba)

    pred_stack = np.vstack(all_preds)
    n_samples = pred_stack.shape[1]

    if strategy == "vote":
        combined_pred = np.round(pred_stack.mean(axis=0)).astype(bool)
        combined_proba = None
        if all_probas:
            combined_proba = np.vstack(all_probas).mean(axis=0)
    elif strategy == "average_proba":
        if not all_probas:
            raise ValueError("average_proba requires models with predict_proba support.")
        combined_proba = np.vstack(all_probas).mean(axis=0)
        combined_pred = combined_proba >= threshold
    elif strategy == "any_positive":
        combined_pred = pred_stack.any(axis=0)
        combined_proba = np.vstack(all_probas).mean(axis=0) if all_probas else None
    elif strategy == "all_positive":
        combined_pred = pred_stack.all(axis=0)
        combined_proba = np.vstack(all_probas).mean(axis=0) if all_probas else None
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {
        "model_names": names,
        "source": source,
        "strategy": strategy,
        "predictions": combined_pred,
        "positive_probabilities": combined_proba,
        "individual_predictions": {name: preds for name, preds in zip(names, all_preds)},
        "individual_positive_probabilities": {
            name: probas for name, probas in zip(names, all_probas)
        }
        if all_probas
        else {},
    }


# Backward-compatible helpers for single-model bundles
def save_sklearn_classifier(
    pipeline,
    model_name: str,
    feature_names: list[str],
    export_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    return save_sklearn_model_suite(
        models_full={model_name: pipeline},
        feature_names=feature_names,
        export_path=export_path,
        metadata=metadata,
    )


def load_sklearn_classifier(export_path: str | Path) -> dict[str, Any]:
    return load_sklearn_model_suite(export_path)


def predict_classifier(bundle: dict[str, Any], features: np.ndarray, model_name: str | None = None) -> np.ndarray:
    if model_name is None:
        if "pipeline" in bundle:
            pipeline = bundle["pipeline"]
            preds, _ = _predict_one_pipeline(pipeline, features)
            return preds
        model_name = list_models(bundle)[0]
    return predict_with_models(bundle, [model_name], features)["predictions"]


def predict_classifier_proba(
    bundle: dict[str, Any],
    features: np.ndarray,
    model_name: str | None = None,
) -> np.ndarray:
    if model_name is None:
        if "pipeline" in bundle:
            _, probas = _predict_one_pipeline(bundle["pipeline"], features)
            if probas is None:
                raise ValueError("Model does not support predict_proba.")
            return probas
        model_name = list_models(bundle)[0]
    result = predict_with_models(bundle, [model_name], features, strategy="average_proba")
    probas = result["positive_probabilities"]
    if probas is None:
        raise ValueError(f"Model '{model_name}' does not support predict_proba.")
    return probas


def clone_unfitted_estimator(estimator):
    """Create a fresh estimator with the same hyperparameters."""
    return deepcopy(estimator)
