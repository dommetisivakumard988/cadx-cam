"""
CadX Studio — Feeds & Speeds FastAPI endpoint
Wraps feeds_speeds_db.py lookup table and exposes an ML-ready skeleton.

Endpoint: POST /cutting-parameters
Also exposes: GET /cutting-parameters/materials  (list all known materials)
              GET /cutting-parameters/operations  (list valid operation types)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from feeds_speeds_db import (
    MATERIAL_ALIASES,
    CuttingParams,
    get_params,
)

router = APIRouter(prefix="/cutting-parameters", tags=["feeds-speeds"])


# ── Request / response models ──────────────────────────────────────

class CuttingParamsRequest(BaseModel):
    material:         str   = Field(...,   description="Material ID e.g. 'en8', 'al6061', 'ss304'")
    operation:        str   = Field(...,   description="'rough', 'finish', or 'drill'")
    tool_dia_mm:      float = Field(...,   ge=1.0, le=100.0)
    machine_max_rpm:  int   = Field(6000,  ge=500, le=60000)
    num_flutes:       int   = Field(4,     ge=2,   le=8,    description="Number of cutter flutes")
    use_ml:           bool  = Field(False, description="Use ML model if available (falls back to lookup)")

    @validator("operation")
    def validate_operation(cls, v: str) -> str:
        allowed = {"rough", "finish", "drill"}
        if v.lower() not in allowed:
            raise ValueError(f"operation must be one of {allowed}")
        return v.lower()

    @validator("material")
    def validate_material(cls, v: str) -> str:
        key = v.lower().replace(" ", "_")
        if key not in MATERIAL_ALIASES:
            known = sorted(set(MATERIAL_ALIASES.values()))
            raise ValueError(f"Unknown material '{v}'. Known IDs: {known}")
        return key


@dataclass
class CuttingParamsResponse:
    # Core output
    material:         str
    material_label:   str
    operation:        str
    tool_dia_mm:      float
    num_flutes:       int

    # Calculated parameters
    spindle_rpm:      int
    feed_mmmin:       float
    plunge_mmmin:     float
    depth_of_cut_mm:  float
    stepover_mm:      float
    stepover_pct:     float
    chipload_mm:      float
    surface_speed_mmin: float   # SFM converted to m/min

    # Machine info
    machine_max_rpm:  int
    rpm_clamped:      bool      # True if RPM was clamped to machine limit

    # Quality / confidence
    source:           str       # "lookup_table" | "ml_model" | "ml_fallback"
    confidence:       float     # 0.0–1.0

    # Derived recommendations
    recommended_coolant:  str
    cycle_time_factor:    float  # relative — higher = longer cycle time
    tool_life_warning:    Optional[str] = None
    notes:                List[str] = field(default_factory=list)


# ── Material display labels ────────────────────────────────────────

MATERIAL_LABELS: Dict[str, str] = {
    "al6061": "Aluminium 6061-T6",
    "en8":    "EN8 Mild Steel",
    "en24":   "EN24 Alloy Steel",
    "ss304":  "Stainless Steel 304",
    "brass":  "Brass C360",
}

COOLANT_MAP: Dict[str, str] = {
    "al6061": "Flood or mist — prevents built-up edge",
    "en8":    "Flood coolant — moderate heat generation",
    "en24":   "Flood coolant — high heat, use cutting oil on final pass",
    "ss304":  "Heavy flood + high-pressure coolant — work hardening risk",
    "brass":  "Dry or light mist — brass machines well without coolant",
}


# ── ML skeleton ───────────────────────────────────────────────────

class FeedSpeedMLModel:
    """
    Skeleton for a physics-informed ML model.
    Currently a stub that falls back to the lookup table.

    To upgrade:
      1. Collect machining data (material, tool_dia, operation, measured
         actual_feed, actual_rpm, surface_finish_Ra, tool_life_passes).
      2. Train a GradientBoostingRegressor (scikit-learn) or small MLP
         (PyTorch) on this data.
      3. Save the model: joblib.dump(model, 'models/feeds_speeds_gb.joblib')
      4. Load it in __init__ and call predict() in calculate().

    Feature vector (8 features):
      [material_hardness_HRC, tool_dia_mm, num_flutes,
       is_rough, is_finish, is_drill,
       machine_rpm_fraction,   # tool_rpm / machine_max_rpm
       tool_overhang_ratio]    # length/dia (fixed to 3.0 for now)
    """

    # Approximate Rockwell hardness for feature engineering
    MATERIAL_HRC: Dict[str, float] = {
        "al6061": 15.0,
        "en8":    20.0,
        "en24":   35.0,
        "ss304":  25.0,
        "brass":  10.0,
    }

    def __init__(self) -> None:
        self._model      = None
        self._model_path = Path(__file__).parent / "models" / "feeds_speeds_gb.joblib"
        self._load_model()

    def _load_model(self) -> None:
        if not self._model_path.exists():
            return
        try:
            import joblib
            self._model = joblib.load(self._model_path)
            print(f"[ML] Loaded feeds/speeds model from {self._model_path}")
        except Exception as e:
            print(f"[ML] Could not load model: {e}")

    @property
    def available(self) -> bool:
        return self._model is not None

    def _build_features(
        self,
        material:   str,
        operation:  str,
        tool_dia:   float,
        num_flutes: int,
        max_rpm:    int,
    ) -> list:
        hrc     = self.MATERIAL_HRC.get(material, 20.0)
        is_r    = 1.0 if operation == "rough"  else 0.0
        is_f    = 1.0 if operation == "finish" else 0.0
        is_d    = 1.0 if operation == "drill"  else 0.0
        rpm_frac = min(1.0, 6000 / max(max_rpm, 1))
        return [hrc, tool_dia, num_flutes, is_r, is_f, is_d, rpm_frac, 3.0]

    def predict(
        self,
        material:        str,
        operation:       str,
        tool_dia_mm:     float,
        num_flutes:      int,
        machine_max_rpm: int,
    ) -> Optional[CuttingParams]:
        """
        Run ML inference. Returns None if model unavailable or prediction fails.
        """
        if not self.available:
            return None
        try:
            import numpy as np
            feats = np.array([self._build_features(
                material, operation, tool_dia_mm, num_flutes, machine_max_rpm,
            )])
            pred = self._model.predict(feats)[0]   # [spindle, feed, plunge, doc, stepover]
            return CuttingParams(
                spindle_rpm  = int(min(pred[0], machine_max_rpm)),
                feed_mmmin   = round(max(10.0, pred[1]), 1),
                plunge_mmmin = round(max(5.0,  pred[2]), 1),
                depth_of_cut = round(max(0.1,  pred[3]), 2),
                stepover_pct = round(min(0.8, max(0.05, pred[4])), 3),
                chipload_mm  = round(max(0.005, pred[1] / max(pred[0], 1) / num_flutes), 4),
            )
        except Exception as e:
            print(f"[ML] Inference error: {e}")
            return None

    # ── Training skeleton ────────────────────────────────────────

    @staticmethod
    def train(data_csv_path: str) -> None:
        """
        Train a GradientBoostingRegressor on collected machining data.
        CSV columns:
          material, operation, tool_dia_mm, num_flutes, max_rpm,
          actual_spindle_rpm, actual_feed_mmmin, actual_plunge_mmmin,
          actual_doc_mm, actual_stepover_pct

        Run: python -c "from feeds_speeds_api import FeedSpeedMLModel; FeedSpeedMLModel.train('data/feeds.csv')"
        """
        import csv
        import joblib
        import numpy as np
        from sklearn.ensemble              import GradientBoostingRegressor
        from sklearn.multioutput           import MultiOutputRegressor
        from sklearn.model_selection       import train_test_split
        from sklearn.metrics               import mean_absolute_error

        rows = []
        with open(data_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mat  = MATERIAL_ALIASES.get(row["material"].lower(), "en8")
                hrc  = FeedSpeedMLModel.MATERIAL_HRC.get(mat, 20.0)
                op   = row["operation"].lower()
                is_r = 1.0 if op == "rough"  else 0.0
                is_f = 1.0 if op == "finish" else 0.0
                is_d = 1.0 if op == "drill"  else 0.0
                dia  = float(row["tool_dia_mm"])
                flu  = int(row["num_flutes"])
                mrpm = int(row["max_rpm"])

                X_row = [hrc, dia, flu, is_r, is_f, is_d, 6000 / max(mrpm, 1), 3.0]
                y_row = [
                    float(row["actual_spindle_rpm"]),
                    float(row["actual_feed_mmmin"]),
                    float(row["actual_plunge_mmmin"]),
                    float(row["actual_doc_mm"]),
                    float(row["actual_stepover_pct"]),
                ]
                rows.append((X_row, y_row))

        X = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        base  = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
        model = MultiOutputRegressor(base, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        print(f"[ML] Training complete — MAE: {mae:.4f}")

        out_path = Path(__file__).parent / "models" / "feeds_speeds_gb.joblib"
        out_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, out_path)
        print(f"[ML] Model saved to {out_path}")


# ── Singleton ML model ─────────────────────────────────────────────
_ml_model = FeedSpeedMLModel()


# ── Core calculation function ──────────────────────────────────────

def calculate_cutting_params(req: CuttingParamsRequest) -> CuttingParamsResponse:
    """
    Returns cutting parameters — from ML model if available and requested,
    otherwise from the lookup table.
    """
    mat_id   = MATERIAL_ALIASES.get(req.material, "en8")
    source   = "lookup_table"
    confidence = 0.92

    # Try ML model first if requested
    params: Optional[CuttingParams] = None
    if req.use_ml and _ml_model.available:
        params = _ml_model.predict(
            mat_id, req.operation, req.tool_dia_mm,
            req.num_flutes, req.machine_max_rpm,
        )
        if params is not None:
            source     = "ml_model"
            confidence = 0.85   # lower confidence until model is well-trained
        else:
            source = "ml_fallback"

    # Fall back to lookup table
    if params is None:
        params = get_params(mat_id, req.operation, req.tool_dia_mm, req.machine_max_rpm)

    rpm_clamped = params.spindle_rpm >= req.machine_max_rpm

    # Surface speed
    surface_speed = (math.pi * req.tool_dia_mm * params.spindle_rpm) / 1000

    # Adjusted chipload per flute (recalculate from final feed/rpm)
    chipload = params.feed_mmmin / (max(params.spindle_rpm, 1) * req.num_flutes)

    # Stepover in mm
    stepover_mm = round(req.tool_dia_mm * params.stepover_pct, 3)

    # Build notes
    notes: List[str] = []
    if rpm_clamped:
        notes.append(
            f"Spindle RPM clamped to machine limit ({req.machine_max_rpm} RPM). "
            "Feed reduced proportionally."
        )
    if req.operation == "drill" and req.tool_dia_mm < 5.0:
        notes.append(
            "Small drill (< Ø5mm) — reduce feed by 20% on first hole to check runout. "
            "Use peck drilling (G83) for depths > 3×D."
        )
    if req.operation == "finish" and mat_id == "ss304":
        notes.append(
            "Stainless finishing: take a light spring pass (0.05mm) at same speed "
            "to achieve Ra < 1.6 μm surface finish."
        )
    if mat_id == "en24" and req.operation == "rough":
        notes.append(
            "EN24 work-hardens — keep tool engaged. Avoid rubbing/dwelling. "
            "Use TiAlN-coated carbide tooling."
        )

    # Tool life warning
    tool_life_warning: Optional[str] = None
    if chipload > 0.08:
        tool_life_warning = (
            f"Chipload {chipload:.4f}mm/tooth is high — monitor tool wear closely. "
            "Consider reducing feed by 10–15% for longer tool life."
        )
    elif chipload < 0.005:
        tool_life_warning = (
            f"Chipload {chipload:.4f}mm/tooth is very low — risk of rubbing rather than cutting. "
            "Increase feed or reduce RPM."
        )

    # Cycle time factor (relative — higher = slower)
    ctf = round(1.0 / (params.feed_mmmin * params.depth_of_cut * stepover_mm + 1e-6), 8)
    ctf_normalised = min(10.0, ctf * 1e5)   # scale to human-readable range

    return CuttingParamsResponse(
        material          = mat_id,
        material_label    = MATERIAL_LABELS.get(mat_id, mat_id.upper()),
        operation         = req.operation,
        tool_dia_mm       = req.tool_dia_mm,
        num_flutes        = req.num_flutes,
        spindle_rpm       = params.spindle_rpm,
        feed_mmmin        = params.feed_mmmin,
        plunge_mmmin      = params.plunge_mmmin,
        depth_of_cut_mm   = params.depth_of_cut,
        stepover_mm       = stepover_mm,
        stepover_pct      = params.stepover_pct,
        chipload_mm       = round(chipload, 5),
        surface_speed_mmin = round(surface_speed, 1),
        machine_max_rpm   = req.machine_max_rpm,
        rpm_clamped       = rpm_clamped,
        source            = source,
        confidence        = confidence,
        recommended_coolant = COOLANT_MAP.get(mat_id, "Flood coolant"),
        cycle_time_factor = round(ctf_normalised, 3),
        tool_life_warning = tool_life_warning,
        notes             = notes,
    )


# ── FastAPI routes ─────────────────────────────────────────────────

@router.post("", response_model=None)
async def get_cutting_parameters(req: CuttingParamsRequest):
    """
    Calculate optimal feeds and speeds for a given material,
    operation type, and tool diameter.
    """
    try:
        result = calculate_cutting_params(req)
        return asdict(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {e}")


@router.get("/materials")
async def list_materials():
    """Return all supported material IDs and their display labels."""
    return {
        "materials": [
            {"id": mat_id, "label": MATERIAL_LABELS.get(mat_id, mat_id.upper())}
            for mat_id in sorted(set(MATERIAL_ALIASES.values()))
        ]
    }


@router.get("/operations")
async def list_operations():
    """Return valid operation types."""
    return {
        "operations": [
            {"id": "rough",  "label": "Pocket roughing",    "description": "High MRR material removal"},
            {"id": "finish", "label": "Contour finishing",  "description": "Low DOC, low stepover surface pass"},
            {"id": "drill",  "label": "Drilling cycle",     "description": "G81/G83 drill cycle parameters"},
        ]
    }