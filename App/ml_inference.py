"""ML inference helpers for the app.

Loads sklearn classifier bundles exported by ``model/ml_model_io.py``,
extracts the same 13-feature vector used at training time from the app's
``delta`` entry, and produces a POS / NEG label for one or several models.

Feature order (must match the training notebook
``model/Persuade_ESC_2026.ipynb`` -> ``Read.init``):

    SR_Duration, S1_Duration, S2_Duration, S3_Duration,
    S1_n Deflection, S2_n Deflection, S3_n Deflection,
    S1_Delta, S2_Delta, S3_Delta,
    S1_Voltage, S2_Voltage, S3_Voltage
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "SR_Duration",
    "S1_Duration",
    "S2_Duration",
    "S3_Duration",
    "S1_n Deflection",
    "S2_n Deflection",
    "S3_n Deflection",
    "S1_Delta",
    "S2_Delta",
    "S3_Delta",
    "S1_Voltage",
    "S2_Voltage",
    "S3_Voltage",
)

# Same quality gate as ``LAT_points`` in the notebook: under this minimum
# the point was excluded from training, so predicting on it would be OOD.
MIN_STIM_VOLTAGE = 0.05

SelectionStrategy = Literal["vote", "average_proba", "any_positive", "all_positive"]


def _as_2d(features) -> np.ndarray:
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_window(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_is_number(v) for v in value)
    )


def _pick_sinus_window(stim: list, sinus: list) -> list | None:
    """Pick the sinus window matching the notebook rule.

    Notebook ``LAT_points.point.init`` picks the last sinus strictly before
    the first stim; if every sinus starts after the first stim it falls back
    to the last entry. We walk back across invalid (``False``) entries so a
    failed onset detection on one sinus window doesn't block prediction.
    """
    if not sinus:
        return None
    first_stim_start = int(stim[0][0])
    selected_idx: int | None = None
    for i, entry in enumerate(sinus):
        if _is_window(entry) and int(entry[0]) > first_stim_start:
            selected_idx = i - 1
            break
    if selected_idx is None:
        selected_idx = len(sinus) - 1
    j = min(selected_idx, len(sinus) - 1)
    while j >= 0:
        if _is_window(sinus[j]):
            return list(sinus[j])
        j -= 1
    for entry in sinus:
        if _is_window(entry):
            return list(entry)
    return None


def extract_features_from_c1(c1: dict | None) -> np.ndarray | None:
    """Build the 13-feature vector or return ``None`` if the point is not
    eligible (training-time quality gate failed).
    """
    if not isinstance(c1, dict):
        return None
    stim = c1.get("stim") or []
    sinus = c1.get("sinus") or []
    voltage_stim = c1.get("voltage_stim") or []
    deflection_stim = c1.get("deflection_stim") or []
    if len(stim) < 3 or len(voltage_stim) < 3 or len(deflection_stim) < 3:
        return None
    if not all(_is_window(s) for s in stim[:3]):
        return None
    if not all(_is_number(v) for v in voltage_stim[:3]):
        return None
    if not all(_is_number(d) for d in deflection_stim[:3]):
        return None
    sr_window = _pick_sinus_window(stim, sinus)
    if sr_window is None:
        return None
    sr_dur = float(int(sr_window[1]) - int(sr_window[0]))
    s1_dur = float(int(stim[0][1]) - int(stim[0][0]))
    s2_dur = float(int(stim[1][1]) - int(stim[1][0]))
    s3_dur = float(int(stim[2][1]) - int(stim[2][0]))
    s1_def, s2_def, s3_def = (float(deflection_stim[i]) for i in range(3))
    s1_v, s2_v, s3_v = (float(voltage_stim[i]) for i in range(3))
    if min(s1_v, s2_v, s3_v) < MIN_STIM_VOLTAGE:
        return None
    return np.asarray(
        [
            sr_dur,
            s1_dur,
            s2_dur,
            s3_dur,
            s1_def,
            s2_def,
            s3_def,
            s1_dur - sr_dur,
            s2_dur - sr_dur,
            s3_dur - sr_dur,
            s1_v,
            s2_v,
            s3_v,
        ],
        dtype=float,
    )


def extract_features_from_delta_entry(delta_entry) -> np.ndarray | None:
    """``delta_entry`` is either ``0`` (uncomputed) or ``[pnum, label, c1]``."""
    if not isinstance(delta_entry, (list, tuple)) or len(delta_entry) < 3:
        return None
    return extract_features_from_c1(delta_entry[2])


class MLBundle:
    """Wrap a joblib bundle written by ``model/ml_model_io.py``."""

    def __init__(self, path: str | Path, data: dict[str, Any]):
        self.path = str(path)
        self.data = data

    @property
    def model_names(self) -> list[str]:
        names = self.data.get("model_names")
        if names:
            return list(names)
        return list((self.data.get("models_full") or {}).keys())

    @property
    def feature_names(self) -> list[str]:
        return list(self.data.get("feature_names") or [])

    @property
    def cv_summary(self) -> list[dict[str, Any]]:
        return list(self.data.get("cv_summary") or [])

    def _pipeline(self, model_name: str):
        try:
            return self.data["models_full"][model_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown model '{model_name}'. Available: {self.model_names}"
            ) from exc

    def predict(
        self,
        features,
        model_names: Iterable[str],
        *,
        strategy: SelectionStrategy = "vote",
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        names = list(model_names)
        if not names:
            raise ValueError("model_names must contain at least one name.")
        X = _as_2d(features)
        all_pred: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []
        for name in names:
            pipe = self._pipeline(name)
            pred = pipe.predict(X)
            all_pred.append(np.asarray(pred))
            if hasattr(pipe, "predict_proba"):
                try:
                    all_proba.append(pipe.predict_proba(X)[:, 1])
                except Exception:
                    traceback.print_exc()
        pred_stack = np.vstack(all_pred)
        if strategy == "vote":
            combined_pred = np.round(pred_stack.mean(axis=0)).astype(bool)
            combined_proba = np.vstack(all_proba).mean(axis=0) if all_proba else None
        elif strategy == "average_proba":
            if not all_proba:
                raise ValueError("average_proba requires predict_proba support.")
            combined_proba = np.vstack(all_proba).mean(axis=0)
            combined_pred = combined_proba >= threshold
        elif strategy == "any_positive":
            combined_pred = pred_stack.any(axis=0)
            combined_proba = np.vstack(all_proba).mean(axis=0) if all_proba else None
        elif strategy == "all_positive":
            combined_pred = pred_stack.all(axis=0)
            combined_proba = np.vstack(all_proba).mean(axis=0) if all_proba else None
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return {
            "predictions": combined_pred,
            "positive_probabilities": combined_proba,
        }


def load_ml_bundle(path: str | Path) -> MLBundle:
    """Load a joblib bundle and wrap it. ``joblib`` is imported lazily so
    the app can still start when it isn't installed in the runtime env.
    """
    import joblib

    data = joblib.load(path)
    return MLBundle(path=path, data=data)


def predicted_label_for_delta(
    bundle: MLBundle,
    model_names: Iterable[str],
    delta_entry,
    *,
    strategy: SelectionStrategy = "vote",
    threshold: float = 0.5,
) -> str:
    """Return ``"POS"`` / ``"NEG"`` for a delta entry, or ``""`` if the
    point is not eligible (no features) or prediction failed.
    """
    features = extract_features_from_delta_entry(delta_entry)
    if features is None:
        return ""
    try:
        result = bundle.predict(
            features, model_names, strategy=strategy, threshold=threshold
        )
    except Exception:
        traceback.print_exc()
        return ""
    pred = result["predictions"]
    if pred is None or len(pred) == 0:
        return ""
    return "POS" if bool(pred[0]) else "NEG"


def find_local_model_files(start_dir: str | Path | None = None) -> list[Path]:
    """Search a sibling ``model/`` folder for ``.joblib`` bundles."""
    base = Path(start_dir).resolve() if start_dir else Path(__file__).resolve().parent
    here = base
    for _ in range(4):
        sibling = here.parent / "model"
        if sibling.is_dir():
            return sorted(sibling.glob("*.joblib"))
        here = here.parent
    return []
