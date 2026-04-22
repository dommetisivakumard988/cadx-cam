"""
CadX Studio — CAM Engine
Reads a STEP file, tessellates geometry, generates toolpaths with OpenCAMLib,
and returns CL (cutter-location) data ready for post-processing.

Supports:
  • Pocket roughing  — adaptive Z-level waterline clearing
  • Contour finishing — boundary-following finish pass
  • Drilling cycles   — detect circular faces, generate drill positions
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import cadquery as cq
import numpy as np
import trimesh

try:
    import ocl                          # OpenCAMLib
    OCL_AVAILABLE = True
except ImportError:
    OCL_AVAILABLE = False
    print("WARNING: opencamlib not installed — using fallback raster scan")

from feeds_speeds_db import get_params, CuttingParams


# ── Data structures ────────────────────────────────────────────────

@dataclass
class CLPoint:
    """Single cutter-location point."""
    x:      float
    y:      float
    z:      float
    rapid:  bool  = False   # True → G00, False → G01/G02/G03
    plunge: bool  = False   # True → use plunge rate
    arc:    Optional[dict] = None  # {'type':'CW'|'CCW','cx':..,'cy':..,'r':..}


@dataclass
class DrillCycle:
    """G81/G83 drilling cycle parameters."""
    x:        float
    y:        float
    z_top:    float   # R plane (clearance)
    z_depth:  float   # final depth
    peck:     bool    = False
    peck_dep: float   = 5.0   # peck depth if peck=True


@dataclass
class Operation:
    name:       str
    op_type:    str        # "rough" | "finish" | "drill"
    tool_no:    int
    tool_dia:   float
    cutting:    CuttingParams
    cl_points:  List[CLPoint]   = field(default_factory=list)
    drills:     List[DrillCycle] = field(default_factory=list)
    comment:    str = ""


@dataclass
class CLData:
    """Complete CL dataset returned to the post-processor."""
    part_name:    str
    material:     str
    bounding_box: dict          # {xmin,xmax,ymin,ymax,zmin,zmax}
    stock_z_top:  float         # top of raw stock
    rapid_z:      float         # safe rapid height above stock
    operations:   List[Operation] = field(default_factory=list)
    warnings:     List[str]       = field(default_factory=list)


# ── STEP reader ────────────────────────────────────────────────────

def load_step(step_path: str) -> tuple[cq.Workplane, dict]:
    """Load STEP file and return CadQuery workplane + bounding box dict."""
    wp  = cq.importers.importStep(step_path)
    bb  = wp.val().BoundingBox()
    box = {
        "xmin": round(bb.xmin, 4), "xmax": round(bb.xmax, 4),
        "ymin": round(bb.ymin, 4), "ymax": round(bb.ymax, 4),
        "zmin": round(bb.zmin, 4), "zmax": round(bb.zmax, 4),
        "xsize": round(bb.xsize, 4),
        "ysize": round(bb.ysize, 4),
        "zsize": round(bb.zsize, 4),
    }
    return wp, box


def step_to_trimesh(wp: cq.Workplane, tolerance: float = 0.05) -> trimesh.Trimesh:
    """Tessellate CadQuery solid to trimesh for OpenCAMLib."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        stl_path = f.name
    try:
        cq.exporters.export(wp.val(), stl_path, tolerance=tolerance, angularTolerance=0.1)
        mesh = trimesh.load(stl_path)
    finally:
        os.unlink(stl_path)
    return mesh


def mesh_to_ocl_stlsurf(mesh: trimesh.Trimesh) -> "ocl.STLSurf":
    """Convert trimesh to OpenCAMLib STLSurf."""
    surf = ocl.STLSurf()
    verts = mesh.vertices
    for face in mesh.faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        t = ocl.Triangle(
            ocl.Point(*v0.tolist()),
            ocl.Point(*v1.tolist()),
            ocl.Point(*v2.tolist()),
        )
        surf.addTriangle(t)
    return surf


# ── Toolpath generators ────────────────────────────────────────────

def _make_cyl_cutter(tool_dia: float) -> "ocl.CylCutter":
    return ocl.CylCutter(tool_dia, 100.0)   # dia, length


def gen_pocket_roughing_ocl(
    surf:    "ocl.STLSurf",
    box:     dict,
    cutter:  "ocl.CylCutter",
    params:  CuttingParams,
) -> List[CLPoint]:
    """
    Adaptive Z-level waterline roughing using OpenCAMLib.
    Returns list of CLPoint objects.
    """
    tool_r    = cutter.diameter / 2
    stepover  = cutter.diameter * params.stepover_pct
    z_top     = box["zmax"]
    z_bot     = box["zmin"]
    n_levels  = max(1, math.ceil((z_top - z_bot) / params.depth_of_cut))
    z_levels  = [z_top - i * params.depth_of_cut for i in range(1, n_levels + 1)]
    z_levels[-1] = z_bot   # ensure we reach the floor

    points: List[CLPoint] = []
    safe_z = z_top + 10.0

    # Start rapid to safe height, first XY
    points.append(CLPoint(box["xmin"], box["ymin"], safe_z, rapid=True))

    for z in z_levels:
        wl = ocl.AdaptiveWaterline()
        wl.setSTL(surf)
        wl.setCutter(cutter)
        wl.setSampling(0.4)
        wl.setZ(z)
        wl.run()

        loops = wl.getLoops()
        if not loops:
            continue

        # Rapid down to clearance plane for this Z level
        first = loops[0][0]
        points.append(CLPoint(first.x, first.y, safe_z, rapid=True))
        points.append(CLPoint(first.x, first.y, z + 2.0, rapid=True))
        points.append(CLPoint(first.x, first.y, z, plunge=True))

        for loop in loops:
            if not loop:
                continue
            p0 = loop[0]
            # Move to loop start
            points.append(CLPoint(p0.x, p0.y, safe_z, rapid=True))
            points.append(CLPoint(p0.x, p0.y, z + 2.0, rapid=True))
            points.append(CLPoint(p0.x, p0.y, z, plunge=True))
            for pt in loop[1:]:
                points.append(CLPoint(round(pt.x, 4), round(pt.y, 4), round(z, 4)))
            # Close loop
            points.append(CLPoint(round(p0.x, 4), round(p0.y, 4), round(z, 4)))

    # Retract
    points.append(CLPoint(points[-1].x, points[-1].y, safe_z, rapid=True))
    return points


def gen_pocket_roughing_fallback(box: dict, params: CuttingParams, tool_dia: float) -> List[CLPoint]:
    """
    Fallback raster scan when OCL is unavailable.
    Generates a boustrophedon (zigzag) pattern for each Z level.
    """
    stepover = tool_dia * params.stepover_pct
    z_top    = box["zmax"]
    z_bot    = box["zmin"]
    n_levels = max(1, math.ceil((z_top - z_bot) / params.depth_of_cut))
    safe_z   = z_top + 10.0

    points: List[CLPoint] = []
    points.append(CLPoint(box["xmin"], box["ymin"], safe_z, rapid=True))

    for i in range(n_levels):
        z      = max(z_bot, z_top - (i + 1) * params.depth_of_cut)
        y_vals = np.arange(box["ymin"], box["ymax"], stepover)

        for j, y in enumerate(y_vals):
            x_start = box["xmin"] if j % 2 == 0 else box["xmax"]
            x_end   = box["xmax"] if j % 2 == 0 else box["xmin"]

            if j == 0:
                points.append(CLPoint(x_start, y, safe_z, rapid=True))
                points.append(CLPoint(x_start, y, z + 2.0, rapid=True))
                points.append(CLPoint(x_start, y, z, plunge=True))
            else:
                points.append(CLPoint(x_start, round(y, 4), round(z, 4)))

            points.append(CLPoint(round(x_end, 4), round(y, 4), round(z, 4)))

    points.append(CLPoint(points[-1].x, points[-1].y, safe_z, rapid=True))
    return points


def gen_contour_finishing(
    surf:    Optional["ocl.STLSurf"],
    box:     dict,
    params:  CuttingParams,
    tool_dia: float,
) -> List[CLPoint]:
    """
    Contour finishing pass: drop-cutter scan along boundary,
    then a full-surface drop-cutter pass for 3D finishing.
    """
    safe_z   = box["zmax"] + 10.0
    points: List[CLPoint] = []

    if OCL_AVAILABLE and surf is not None:
        cutter  = _make_cyl_cutter(tool_dia)
        pdc     = ocl.PathDropCutter()
        pdc.setSTL(surf)
        pdc.setCutter(cutter)
        pdc.setSampling(0.2)
        pdc.setMinSampling(0.01)

        # Build a grid of paths
        stepover = tool_dia * params.stepover_pct
        y_vals   = np.arange(box["ymin"], box["ymax"] + stepover, stepover)
        for j, y in enumerate(y_vals):
            path = ocl.Path()
            x0   = box["xmin"] - tool_dia
            x1   = box["xmax"] + tool_dia
            path.append(ocl.Line(ocl.Point(x0, y, 0), ocl.Point(x1, y, 0)))
            pdc.setPath(path)
            pdc.run()
            cl = pdc.getCLPoints()

            if not cl:
                continue
            # Approach
            points.append(CLPoint(cl[0].x, cl[0].y, safe_z, rapid=True))
            points.append(CLPoint(cl[0].x, cl[0].y, cl[0].z + 2.0, rapid=True))
            points.append(CLPoint(cl[0].x, cl[0].y, cl[0].z, plunge=True))
            for pt in cl[1:]:
                points.append(CLPoint(round(pt.x, 4), round(pt.y, 4), round(pt.z, 4)))

    else:
        # Fallback: simple Z-constant boundary contour
        margin   = tool_dia / 2
        z        = box["zmax"]
        corners  = [
            (box["xmin"] + margin, box["ymin"] + margin),
            (box["xmax"] - margin, box["ymin"] + margin),
            (box["xmax"] - margin, box["ymax"] - margin),
            (box["xmin"] + margin, box["ymax"] - margin),
        ]
        points.append(CLPoint(corners[0][0], corners[0][1], safe_z, rapid=True))
        points.append(CLPoint(corners[0][0], corners[0][1], z + 2.0, rapid=True))
        points.append(CLPoint(corners[0][0], corners[0][1], z, plunge=True))
        for cx, cy in corners[1:] + [corners[0]]:
            points.append(CLPoint(round(cx, 4), round(cy, 4), round(z, 4)))

    if points:
        points.append(CLPoint(points[-1].x, points[-1].y, safe_z, rapid=True))
    return points


def detect_drill_positions(wp: cq.Workplane, min_dia: float = 3.0, max_dia: float = 40.0) -> List[DrillCycle]:
    """
    Detect circular/cylindrical faces in the STEP model and extract drill positions.
    Returns a list of DrillCycle objects (one per hole).
    """
    drills: List[DrillCycle] = []
    shape = wp.val()
    bb    = shape.BoundingBox()
    z_top = bb.zmax
    safe_z = z_top + 5.0

    for face in shape.Faces():
        if face.geomType() == "CYLINDER":
            try:
                # Get the axis and radius of the cylindrical face
                surf   = face.surface
                center = face.Center()
                # CadQuery exposes radius via _geomAdaptor for cylinders
                adaptor = face._geomAdaptor()
                radius  = adaptor.Radius()
                dia     = radius * 2

                if min_dia <= dia <= max_dia:
                    # Check if face is vertical (axis ≈ Z)
                    axis = adaptor.Axis().Direction()
                    if abs(axis.Z()) > 0.9:   # mostly Z-aligned
                        z_min = face.BoundingBox().zmin
                        drills.append(DrillCycle(
                            x       = round(center.x, 4),
                            y       = round(center.y, 4),
                            z_top   = round(safe_z, 4),
                            z_depth = round(z_min - 1.0, 4),   # 1mm below face bottom
                            peck    = dia < 8,                   # peck for small holes
                            peck_dep = round(dia * 0.5, 2),
                        ))
            except Exception:
                pass   # Skip faces that don't expose geometry cleanly

    # Deduplicate by XY position (within 0.5mm)
    unique: List[DrillCycle] = []
    for d in drills:
        if not any(
            abs(u.x - d.x) < 0.5 and abs(u.y - d.y) < 0.5
            for u in unique
        ):
            unique.append(d)

    return unique


# ── Main entry point ───────────────────────────────────────────────

def generate_toolpath(
    step_path:       str,
    material:        str       = "en8",
    operations_req:  List[str] = None,   # ["rough","finish","drill"]
    tool_dia_rough:  float     = 16.0,
    tool_dia_finish: float     = 10.0,
    tool_dia_drill:  float     = None,   # auto-detected from holes
    machine_max_rpm: int       = 6000,
) -> CLData:
    """
    Main entry point. Returns a CLData object with all operations.
    """
    if operations_req is None:
        operations_req = ["rough", "finish", "drill"]

    # Load geometry
    wp, box = load_step(step_path)
    part_name = Path(step_path).stem
    warnings:  List[str] = []

    # Tessellate for OCL
    surf = None
    ocl_surf = None
    if OCL_AVAILABLE:
        try:
            mesh     = step_to_trimesh(wp, tolerance=0.05)
            ocl_surf = mesh_to_ocl_stlsurf(mesh)
        except Exception as e:
            warnings.append(f"Tessellation warning: {e}. Using fallback paths.")

    stock_z_top = box["zmax"] + 2.0   # 2mm stock allowance on top
    rapid_z     = stock_z_top + 15.0

    cl_data = CLData(
        part_name    = part_name,
        material     = material,
        bounding_box = box,
        stock_z_top  = round(stock_z_top, 4),
        rapid_z      = round(rapid_z, 4),
        warnings     = warnings,
    )

    op_idx = 1

    # ── Pocket roughing ────────────────────────────────────────────
    if "rough" in operations_req:
        params  = get_params(material, "rough", tool_dia_rough, machine_max_rpm)
        cutter  = _make_cyl_cutter(tool_dia_rough) if OCL_AVAILABLE else None

        if OCL_AVAILABLE and ocl_surf and cutter:
            pts = gen_pocket_roughing_ocl(ocl_surf, box, cutter, params)
        else:
            pts = gen_pocket_roughing_fallback(box, params, tool_dia_rough)

        cl_data.operations.append(Operation(
            name      = f"Pocket Roughing — Ø{tool_dia_rough}mm",
            op_type   = "rough",
            tool_no   = op_idx,
            tool_dia  = tool_dia_rough,
            cutting   = params,
            cl_points = pts,
            comment   = f"Adaptive waterline clearing | {material.upper()} | DOC={params.depth_of_cut}mm | Feed={params.feed_mmmin}mm/min",
        ))
        op_idx += 1

    # ── Contour finishing ──────────────────────────────────────────
    if "finish" in operations_req:
        params = get_params(material, "finish", tool_dia_finish, machine_max_rpm)
        pts    = gen_contour_finishing(ocl_surf, box, params, tool_dia_finish)

        cl_data.operations.append(Operation(
            name      = f"Contour Finishing — Ø{tool_dia_finish}mm",
            op_type   = "finish",
            tool_no   = op_idx,
            tool_dia  = tool_dia_finish,
            cutting   = params,
            cl_points = pts,
            comment   = f"Drop-cutter surface finishing | Stepover={params.stepover_pct*100:.0f}% | Feed={params.feed_mmmin}mm/min",
        ))
        op_idx += 1

    # ── Drilling cycles ────────────────────────────────────────────
    if "drill" in operations_req:
        try:
            drill_cycles = detect_drill_positions(wp)
        except Exception as e:
            warnings.append(f"Drill detection error: {e}")
            drill_cycles = []

        if drill_cycles:
            # Group by diameter (use closest match from known drill sizes)
            # For simplicity, assume all detected holes use same drill
            avg_dia     = tool_dia_drill or 8.0
            drill_params = get_params(material, "drill", avg_dia, machine_max_rpm)

            cl_data.operations.append(Operation(
                name     = f"Drilling — Ø{avg_dia}mm",
                op_type  = "drill",
                tool_no  = op_idx,
                tool_dia = avg_dia,
                cutting  = drill_params,
                drills   = drill_cycles,
                comment  = f"{len(drill_cycles)} holes | {material.upper()} | Feed={drill_params.feed_mmmin}mm/min",
            ))
            op_idx += 1
        else:
            warnings.append("No drill positions detected in STEP geometry.")

    return cl_data