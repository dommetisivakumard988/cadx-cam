"""
Material + operation feeds/speeds lookup table for Indian manufacturing.
Values tested on ACE JT-40 (BT40 spindle, 6000 RPM max, 7.5 kW).
"""
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class CuttingParams:
    spindle_rpm:    int
    feed_mmmin:     float
    plunge_mmmin:   float
    depth_of_cut:   float   # mm
    stepover_pct:   float   # fraction of tool diameter (0-1)
    chipload_mm:    float   # mm/tooth

# key: (material_id, operation, tool_dia_mm)
# We interpolate between known diameters
_TABLE: Dict[Tuple[str, str, float], CuttingParams] = {
    # ── Aluminium 6061 ──────────────────────────────────────────────
    ("al6061", "rough", 10): CuttingParams(4500, 900,  200, 3.0, 0.45, 0.050),
    ("al6061", "rough", 16): CuttingParams(3600, 800,  160, 4.0, 0.45, 0.056),
    ("al6061", "finish", 10): CuttingParams(5000, 600,  120, 0.5, 0.10, 0.030),
    ("al6061", "finish", 16): CuttingParams(4200, 550,  110, 0.5, 0.10, 0.033),
    ("al6061", "drill",   8): CuttingParams(2800, 180,  180, 99,  1.00, 0.064),
    ("al6061", "drill",  10): CuttingParams(2400, 150,  150, 99,  1.00, 0.063),

    # ── EN8 Mild Steel ───────────────────────────────────────────────
    ("en8",    "rough", 10): CuttingParams(2800, 350,   80, 2.0, 0.40, 0.031),
    ("en8",    "rough", 16): CuttingParams(2200, 300,   65, 2.5, 0.40, 0.034),
    ("en8",    "finish", 10): CuttingParams(3200, 250,   60, 0.3, 0.08, 0.020),
    ("en8",    "finish", 16): CuttingParams(2800, 220,   55, 0.3, 0.08, 0.020),
    ("en8",    "drill",   8): CuttingParams(1600,  80,   80, 99,  1.00, 0.025),
    ("en8",    "drill",  10): CuttingParams(1400,  70,   70, 99,  1.00, 0.025),

    # ── EN24 Alloy Steel ────────────────────────────────────────────
    ("en24",   "rough", 10): CuttingParams(2200, 280,   60, 1.5, 0.35, 0.032),
    ("en24",   "rough", 16): CuttingParams(1800, 240,   55, 2.0, 0.35, 0.033),
    ("en24",   "finish", 10): CuttingParams(2600, 200,   50, 0.3, 0.07, 0.019),
    ("en24",   "finish", 16): CuttingParams(2200, 180,   45, 0.3, 0.07, 0.020),
    ("en24",   "drill",   8): CuttingParams(1200,  60,   60, 99,  1.00, 0.025),
    ("en24",   "drill",  10): CuttingParams(1000,  50,   50, 99,  1.00, 0.025),

    # ── SS304 Stainless ──────────────────────────────────────────────
    ("ss304",  "rough", 10): CuttingParams(2000, 200,   40, 1.0, 0.30, 0.025),
    ("ss304",  "rough", 16): CuttingParams(1600, 180,   38, 1.5, 0.30, 0.028),
    ("ss304",  "finish", 10): CuttingParams(2400, 150,   35, 0.2, 0.06, 0.016),
    ("ss304",  "finish", 16): CuttingParams(2000, 140,   32, 0.2, 0.06, 0.018),
    ("ss304",  "drill",   8): CuttingParams(1000,  40,   40, 99,  1.00, 0.020),
    ("ss304",  "drill",  10): CuttingParams( 850,  35,   35, 99,  1.00, 0.021),

    # ── Brass C360 ───────────────────────────────────────────────────
    ("brass",  "rough", 10): CuttingParams(4000, 700,  150, 2.5, 0.45, 0.044),
    ("brass",  "rough", 16): CuttingParams(3200, 600,  130, 3.5, 0.45, 0.047),
    ("brass",  "finish", 10): CuttingParams(4500, 500,  100, 0.4, 0.10, 0.028),
    ("brass",  "drill",   8): CuttingParams(2500, 140,  140, 99,  1.00, 0.056),
    ("brass",  "drill",  10): CuttingParams(2100, 120,  120, 99,  1.00, 0.057),
}

MATERIAL_ALIASES = {
    "aluminium_6061": "al6061", "al6061": "al6061", "aluminium": "al6061",
    "en8": "en8", "mild_steel": "en8", "ms": "en8",
    "en24": "en24", "alloy_steel": "en24",
    "ss304": "ss304", "stainless": "ss304", "ss316": "ss304",
    "brass": "brass", "brass_c360": "brass",
}

def get_params(
    material:    str,
    operation:   str,   # "rough" | "finish" | "drill"
    tool_dia_mm: float,
    machine_max_rpm: int = 6000,
) -> CuttingParams:
    """
    Returns cutting parameters for the given material/operation/tool.
    Interpolates between known diameters. Clamps RPM to machine limit.
    """
    mat = MATERIAL_ALIASES.get(material.lower().replace(" ", "_"), "en8")
    op  = operation.lower()[:6]   # rough | finish | drill

    # Find nearest smaller and larger known diameters
    known_dias = sorted({k[2] for k in _TABLE if k[0] == mat and k[1] == op})
    if not known_dias:
        # Fallback to en8 rough
        known_dias = [10]
        mat = "en8"
        op  = "rough"

    if tool_dia_mm <= known_dias[0]:
        p = _TABLE.get((mat, op, known_dias[0]))
    elif tool_dia_mm >= known_dias[-1]:
        p = _TABLE.get((mat, op, known_dias[-1]))
    else:
        # Linear interpolation between two bracket diameters
        lo = max(d for d in known_dias if d <= tool_dia_mm)
        hi = min(d for d in known_dias if d >= tool_dia_mm)
        if lo == hi:
            p = _TABLE[(mat, op, lo)]
        else:
            plo = _TABLE[(mat, op, lo)]
            phi = _TABLE[(mat, op, hi)]
            frac = (tool_dia_mm - lo) / (hi - lo)
            p = CuttingParams(
                spindle_rpm   = int(plo.spindle_rpm + frac*(phi.spindle_rpm - plo.spindle_rpm)),
                feed_mmmin    = round(plo.feed_mmmin + frac*(phi.feed_mmmin - plo.feed_mmmin), 1),
                plunge_mmmin  = round(plo.plunge_mmmin + frac*(phi.plunge_mmmin - plo.plunge_mmmin), 1),
                depth_of_cut  = round(plo.depth_of_cut + frac*(phi.depth_of_cut - plo.depth_of_cut), 2),
                stepover_pct  = round(plo.stepover_pct + frac*(phi.stepover_pct - plo.stepover_pct), 3),
                chipload_mm   = round(plo.chipload_mm + frac*(phi.chipload_mm - plo.chipload_mm), 4),
            )

    if p is None:
        p = _TABLE.get(("en8", "rough", 10), CuttingParams(2000, 300, 60, 2.0, 0.40, 0.030))

    # Clamp RPM to machine limit
    if p.spindle_rpm > machine_max_rpm:
        ratio = machine_max_rpm / p.spindle_rpm
        p = CuttingParams(
            spindle_rpm  = machine_max_rpm,
            feed_mmmin   = round(p.feed_mmmin * ratio, 1),
            plunge_mmmin = round(p.plunge_mmmin * ratio, 1),
            depth_of_cut = p.depth_of_cut,
            stepover_pct = p.stepover_pct,
            chipload_mm  = p.chipload_mm,
        )
    return p