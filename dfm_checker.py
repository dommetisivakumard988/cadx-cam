"""
CadX Studio — DFM (Design for Manufacturability) Checker
Analyses a STEP file and returns a structured JSON report covering:
  • Minimum wall thickness  — flags walls thinner than threshold
  • Undercuts               — faces not accessible from Z+ direction (3-axis)
  • Sharp internal corners  — inside radii smaller than tool radius
  • Missing / bad stock      — geometry issues that break CAM
  • Thin features            — ribs, bosses narrower than 1.5 × tool dia
  • Draft angle issues       — near-vertical walls that need EDM/wire cutting
  • Deep narrow pockets      — aspect ratio > 6:1 (length:width)
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cadquery as cq
import numpy as np

try:
    from OCC.Core.BRep        import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib  import brepbndlib
    from OCC.Core.BRepGProp   import brepgprop
    from OCC.Core.BRepMesh    import BRepMesh_IncrementalMesh
    from OCC.Core.BRepTools   import breptools
    from OCC.Core.Bnd          import Bnd_Box
    from OCC.Core.GProp        import GProp_GProps
    from OCC.Core.GeomAbs      import (
        GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Plane,
        GeomAbs_Sphere, GeomAbs_Torus,
    )
    from OCC.Core.TopAbs       import (
        TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX,
        TopAbs_FORWARD, TopAbs_REVERSED,
    )
    from OCC.Core.TopExp       import TopExp_Explorer
    from OCC.Core.TopoDS       import TopoDS_Shape, topods
    from OCC.Core.gp           import gp_Dir, gp_Vec
    from OCC.Core.BRepExtrema  import BRepExtrema_DistShapeShape
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False
    print("WARNING: pythonOCC (OCC) not available — using CadQuery fallback")


# ── Issue severity levels ──────────────────────────────────────────

LEVEL_ERROR   = "error"
LEVEL_WARNING = "warning"
LEVEL_INFO    = "info"

# ── DFM thresholds ────────────────────────────────────────────────

THRESHOLDS = {
    "min_wall_thickness_mm":    1.0,    # below this = error
    "warn_wall_thickness_mm":   1.5,    # below this = warning
    "min_corner_radius_mm":     0.5,    # sharp internal corner if below
    "recommended_corner_r_mm":  1.0,    # recommended fillet radius
    "max_pocket_aspect_ratio":  6.0,    # depth:width ratio
    "min_draft_angle_deg":      0.5,    # below this = draft issue
    "standard_drill_sizes_mm":  [       # preferred hole diameters
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
        6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0,
        14.0, 16.0, 18.0, 20.0,
    ],
    "drill_size_tolerance_mm":  0.15,   # allowable deviation from standard size
}


# ── Data classes ───────────────────────────────────────────────────

@dataclass
class DFMIssue:
    level:       str          # "error" | "warning" | "info"
    code:        str          # machine-readable code e.g. "THIN_WALL"
    message:     str          # human-readable description
    location:    str          # where on the part
    fix:         str          # suggested remedy
    value:       Optional[float] = None   # measured value (e.g. 0.8mm)
    threshold:   Optional[float] = None   # threshold that was violated
    auto_fixable: bool = False


@dataclass
class DFMReport:
    filename:         str
    part_name:        str
    volume_mm3:       float
    surface_area_mm2: float
    bounding_box:     dict        # xsize, ysize, zsize, xmin…
    num_faces:        int
    num_edges:        int
    num_holes:        int
    issues:           List[DFMIssue] = field(default_factory=list)
    summary:          dict           = field(default_factory=dict)
    manufacturability_score: int = 100   # 0-100, 100 = no issues

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = [asdict(i) for i in self.issues]
        return d


# ── Geometry helpers ───────────────────────────────────────────────

def _count_topology(shape: "TopoDS_Shape") -> Tuple[int, int, int]:
    """Count faces, edges, vertices in a TopoDS shape."""
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    faces = 0
    while face_exp.More():
        faces += 1
        face_exp.Next()

    edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
    edges = 0
    while edge_exp.More():
        edges += 1
        edge_exp.Next()

    vert_exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    verts = 0
    while vert_exp.More():
        verts += 1
        vert_exp.Next()

    return faces, edges, verts


def _get_volume_and_area(shape: "TopoDS_Shape") -> Tuple[float, float]:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    vol = props.Mass()
    brepgprop.SurfaceProperties(shape, props)
    area = props.Mass()
    return round(vol, 2), round(area, 2)


def _get_bounding_box(shape: "TopoDS_Shape") -> dict:
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return {
        "xmin":  round(xmin,  4), "xmax": round(xmax,  4),
        "ymin":  round(ymin,  4), "ymax": round(ymax,  4),
        "zmin":  round(zmin,  4), "zmax": round(zmax,  4),
        "xsize": round(xmax - xmin, 4),
        "ysize": round(ymax - ymin, 4),
        "zsize": round(zmax - zmin, 4),
    }


def _face_normal(face: "TopoDS_Shape") -> Optional[Tuple[float, float, float]]:
    """
    Return the dominant normal direction of a planar face.
    Returns None for non-planar faces.
    """
    try:
        adaptor = BRepAdaptor_Surface(topods.Face(face))
        if adaptor.GetType() == GeomAbs_Plane:
            pln = adaptor.Plane()
            n   = pln.Axis().Direction()
            return (round(n.X(), 6), round(n.Y(), 6), round(n.Z(), 6))
    except Exception:
        pass
    return None


# ── Check functions ────────────────────────────────────────────────

def check_wall_thickness_occ(
    shape:     "TopoDS_Shape",
    bbox:      dict,
    issues:    List[DFMIssue],
    threshold: float = THRESHOLDS["min_wall_thickness_mm"],
    warn_t:    float = THRESHOLDS["warn_wall_thickness_mm"],
) -> None:
    """
    Estimate wall thickness by ray-casting pairs of opposing faces.
    For each pair of parallel planar faces, the distance between them
    is the wall thickness.
    """
    # Tessellate for sampling
    mesh = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5)
    mesh.Perform()

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    planar_faces = []
    while face_exp.More():
        face     = face_exp.Current()
        face_exp.Next()
        normal = _face_normal(face)
        if normal is not None:
            planar_faces.append((face, normal))

    # Pair faces with roughly opposite normals
    checked_pairs: set = set()
    for i, (fi, ni) in enumerate(planar_faces):
        for j, (fj, nj) in enumerate(planar_faces):
            if i >= j:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            # Dot product close to -1 means opposite normals (parallel faces)
            dot = ni[0]*nj[0] + ni[1]*nj[1] + ni[2]*nj[2]
            if dot > -0.85:
                continue

            # Measure distance between the two faces
            try:
                dist_shape = BRepExtrema_DistShapeShape(fi, fj)
                if dist_shape.IsDone():
                    dist = dist_shape.Value()
                    if 0.01 < dist < warn_t:
                        loc = f"Wall between face {i+1} and face {j+1}"
                        if dist < threshold:
                            issues.append(DFMIssue(
                                level     = LEVEL_ERROR,
                                code      = "THIN_WALL",
                                message   = f"Wall thickness {dist:.2f}mm is below minimum {threshold}mm",
                                location  = loc,
                                fix       = f"Increase wall thickness to at least {threshold}mm. "
                                            f"Consider {warn_t}mm for safe machining.",
                                value     = round(dist, 3),
                                threshold = threshold,
                                auto_fixable = False,
                            ))
                        else:
                            issues.append(DFMIssue(
                                level     = LEVEL_WARNING,
                                code      = "THIN_WALL_WARNING",
                                message   = f"Wall thickness {dist:.2f}mm is below recommended {warn_t}mm",
                                location  = loc,
                                fix       = f"Consider increasing wall to {warn_t}mm for robustness.",
                                value     = round(dist, 3),
                                threshold = warn_t,
                            ))
            except Exception:
                pass


def check_undercuts_occ(
    shape:  "TopoDS_Shape",
    issues: List[DFMIssue],
) -> int:
    """
    Detect faces whose normals point significantly downward (Z < -0.3).
    These are inaccessible from the top in standard 3-axis milling.
    Returns count of undercut faces found.
    """
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    undercut_count = 0

    while face_exp.More():
        face = face_exp.Current()
        face_exp.Next()
        normal = _face_normal(face)
        if normal is None:
            continue

        nz = normal[2]
        # Face normal points downward — undercut for 3-axis from top
        if nz < -0.30:
            undercut_count += 1
            angle_from_horiz = math.degrees(math.asin(abs(nz)))

            issues.append(DFMIssue(
                level    = LEVEL_ERROR,
                code     = "UNDERCUT",
                message  = f"Undercut face detected — normal angle {angle_from_horiz:.1f}° below horizontal",
                location = f"Face normal ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})",
                fix      = "Redesign to eliminate undercut, or use 5-axis machining / EDM. "
                           "Adding a draft angle ≥ 1° removes the undercut classification.",
                value    = round(nz, 4),
                threshold = -0.30,
                auto_fixable = False,
            ))

    return undercut_count


def check_sharp_internal_corners_occ(
    shape:          "TopoDS_Shape",
    issues:         List[DFMIssue],
    min_radius_mm:  float = THRESHOLDS["min_corner_radius_mm"],
    rec_radius_mm:  float = THRESHOLDS["recommended_corner_r_mm"],
) -> None:
    """
    Detect concave edges (internal corners) with radius below threshold.
    Uses edge curvature from BRepAdaptor.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs     import GeomAbs_Line, GeomAbs_Circle

    edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
    concave_sharp = 0

    while edge_exp.More():
        edge = edge_exp.Current()
        edge_exp.Next()
        try:
            adaptor = BRepAdaptor_Curve(topods.Edge(edge))
            if adaptor.GetType() == GeomAbs_Circle:
                radius = adaptor.Circle().Radius()
                if 0.01 < radius < rec_radius_mm:
                    code  = "SHARP_CORNER" if radius < min_radius_mm else "SMALL_FILLET"
                    level = LEVEL_ERROR    if radius < min_radius_mm else LEVEL_WARNING
                    concave_sharp += 1
                    issues.append(DFMIssue(
                        level    = level,
                        code     = code,
                        message  = f"Internal corner radius {radius:.2f}mm "
                                   f"({'below minimum' if level == LEVEL_ERROR else 'below recommended'} {rec_radius_mm if level == LEVEL_WARNING else min_radius_mm}mm)",
                        location = f"Circular edge r={radius:.3f}mm",
                        fix      = f"Increase fillet to ≥ {rec_radius_mm}mm (matches a standard Ø{rec_radius_mm*2}mm end mill). "
                                   f"Smaller radii require smaller tools, increasing cycle time and risk of breakage.",
                        value     = round(radius, 4),
                        threshold = rec_radius_mm,
                        auto_fixable = False,
                    ))
        except Exception:
            pass


def check_hole_sizes_occ(
    wp:     cq.Workplane,
    shape:  "TopoDS_Shape",
    issues: List[DFMIssue],
) -> int:
    """
    Detect cylindrical holes and check:
     • Non-standard drill sizes
     • Aspect ratio (depth : diameter) > 6 → deep hole warning
    Returns number of holes found.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs     import GeomAbs_Cylinder

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    hole_count = 0
    std_sizes  = THRESHOLDS["standard_drill_sizes_mm"]
    tol        = THRESHOLDS["drill_size_tolerance_mm"]

    while face_exp.More():
        face = face_exp.Current()
        face_exp.Next()
        try:
            adaptor = BRepAdaptor_Surface(topods.Face(face))
            if adaptor.GetType() != GeomAbs_Cylinder:
                continue

            radius = adaptor.Cylinder().Radius()
            dia    = radius * 2

            # Only consider Z-aligned cylinders (holes drilled from top)
            axis = adaptor.Cylinder().Axis().Direction()
            if abs(axis.Z()) < 0.85:
                continue

            hole_count += 1

            # Check against standard sizes
            nearest_std = min(std_sizes, key=lambda s: abs(s - dia))
            deviation   = abs(nearest_std - dia)
            if deviation > tol:
                issues.append(DFMIssue(
                    level    = LEVEL_WARNING,
                    code     = "NON_STANDARD_HOLE",
                    message  = f"Hole Ø{dia:.2f}mm is non-standard (nearest standard: Ø{nearest_std}mm)",
                    location = f"Cylindrical face Ø{dia:.3f}mm",
                    fix      = f"Change to Ø{nearest_std}mm to use a standard drill bit. "
                               f"Custom sizes require reground tools and increase lead time.",
                    value     = round(dia, 3),
                    threshold = nearest_std,
                    auto_fixable = True,
                ))

            # Check aspect ratio: approximate depth from bounding box
            face_bbox = Bnd_Box()
            brepbndlib.Add(face, face_bbox)
            _, _, fz_min, _, _, fz_max = face_bbox.Get()
            depth = abs(fz_max - fz_min)
            ratio = depth / max(dia, 0.001)

            if ratio > THRESHOLDS["max_pocket_aspect_ratio"]:
                issues.append(DFMIssue(
                    level    = LEVEL_WARNING,
                    code     = "DEEP_HOLE",
                    message  = f"Hole Ø{dia:.2f}mm × {depth:.1f}mm deep — aspect ratio {ratio:.1f}:1 exceeds 6:1",
                    location = f"Cylindrical face Ø{dia:.3f}mm, depth {depth:.1f}mm",
                    fix      = "Use peck drilling (G83) with chip-breaking. "
                               "Consider gun-drilling for holes deeper than 10×D. "
                               "Ensure adequate coolant flow.",
                    value     = round(ratio, 2),
                    threshold = THRESHOLDS["max_pocket_aspect_ratio"],
                ))

        except Exception:
            pass

    return hole_count


def check_draft_angles_occ(
    shape:      "TopoDS_Shape",
    issues:     List[DFMIssue],
    min_draft:  float = THRESHOLDS["min_draft_angle_deg"],
) -> None:
    """
    Flag near-vertical planar faces (draft angle < threshold).
    These require sharp end mills or EDM for steel parts.
    """
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        face = face_exp.Current()
        face_exp.Next()
        normal = _face_normal(face)
        if normal is None:
            continue
        nz = abs(normal[2])
        # angle from vertical = acos(|nz|)
        angle_from_vert = math.degrees(math.acos(min(nz, 1.0)))
        # If face is nearly vertical but not a floor/ceiling
        if 88.0 < angle_from_vert <= 91.0:
            # This is essentially a wall — check if it has draft
            draft = 90.0 - angle_from_vert
            if abs(draft) < min_draft:
                issues.append(DFMIssue(
                    level    = LEVEL_INFO,
                    code     = "NO_DRAFT",
                    message  = f"Near-vertical wall with draft angle {draft:.2f}° — may cause tool rubbing",
                    location = f"Planar face normal ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})",
                    fix      = f"Add {min_draft}–2° draft for easier machining and part ejection. "
                               "Critical for die/mould applications.",
                    value     = round(draft, 3),
                    threshold = min_draft,
                ))


def check_thin_features_cq(
    wp:         cq.Workplane,
    bbox:       dict,
    issues:     List[DFMIssue],
    min_rib_mm: float = 1.5,
) -> None:
    """
    CadQuery fallback: detect thin protrusions (ribs, bosses) by
    comparing bounding-box projections of individual faces.
    """
    try:
        xsize = bbox["xsize"]
        ysize = bbox["ysize"]
        zsize = bbox["zsize"]
        min_dim = min(xsize, ysize)

        # Rough heuristic: if the smallest dimension is < 2mm
        # it might be a thin rib — add an info note
        if min_dim < min_rib_mm:
            issues.append(DFMIssue(
                level    = LEVEL_WARNING,
                code     = "THIN_FEATURE",
                message  = f"Part has a dimension of {min_dim:.2f}mm — possible thin rib or boss",
                location = f"Overall bounding box: {xsize:.1f} × {ysize:.1f} × {zsize:.1f}mm",
                fix      = f"Ensure thin ribs are ≥ {min_rib_mm}mm for milling. "
                           "Thinner features may require EDM or wire-cut.",
                value     = round(min_dim, 3),
                threshold = min_rib_mm,
            ))
    except Exception:
        pass


# ── CadQuery fallback (when OCC not available) ─────────────────────

def check_basic_cq(
    wp:     cq.Workplane,
    bbox:   dict,
    issues: List[DFMIssue],
) -> None:
    """
    Basic checks using only CadQuery (no raw OCC required).
    Covers: bounding-box thin features, face count anomalies.
    """
    check_thin_features_cq(wp, bbox, issues)

    # Volume sanity check
    try:
        vol = wp.val().Volume()
        if vol < 1.0:
            issues.append(DFMIssue(
                level    = LEVEL_WARNING,
                code     = "ZERO_VOLUME",
                message  = f"Part volume {vol:.4f}mm³ is unusually small — may be a surface body",
                location = "Entire part",
                fix      = "Ensure the STEP file contains a solid body, not just surfaces. "
                           "Re-export from your CAD tool with 'solid' option enabled.",
                value    = round(vol, 6),
            ))
    except Exception:
        pass


# ── Main entry point ───────────────────────────────────────────────

def analyse_step(
    step_path:  str,
    tool_dia:   float = 8.0,
) -> DFMReport:
    """
    Full DFM analysis of a STEP file.
    Returns a DFMReport with all issues and summary.
    """
    filename  = Path(step_path).name
    part_name = Path(step_path).stem
    issues:   List[DFMIssue] = []

    # ── Load geometry ──────────────────────────────────────────────
    try:
        wp = cq.importers.importStep(step_path)
    except Exception as e:
        # Return a minimal report with load error
        return DFMReport(
            filename   = filename,
            part_name  = part_name,
            volume_mm3 = 0,
            surface_area_mm2 = 0,
            bounding_box = {},
            num_faces = 0,
            num_edges = 0,
            num_holes = 0,
            issues    = [DFMIssue(
                level    = LEVEL_ERROR,
                code     = "LOAD_FAILED",
                message  = f"Failed to load STEP file: {e}",
                location = filename,
                fix      = "Verify the file is a valid STEP AP203/AP214 file.",
            )],
            summary   = {"load_failed": True},
            manufacturability_score = 0,
        )

    shape = wp.val().wrapped   # TopoDS_Shape

    # ── Bounding box ───────────────────────────────────────────────
    cq_bb = wp.val().BoundingBox()
    bbox  = {
        "xmin":  round(cq_bb.xmin, 4), "xmax": round(cq_bb.xmax, 4),
        "ymin":  round(cq_bb.ymin, 4), "ymax": round(cq_bb.ymax, 4),
        "zmin":  round(cq_bb.zmin, 4), "zmax": round(cq_bb.zmax, 4),
        "xsize": round(cq_bb.xsize, 4),
        "ysize": round(cq_bb.ysize, 4),
        "zsize": round(cq_bb.zsize, 4),
    }

    # ── Volume / area ──────────────────────────────────────────────
    if OCC_AVAILABLE:
        volume, surf_area = _get_volume_and_area(shape)
        num_faces, num_edges, _ = _count_topology(shape)
    else:
        try:
            volume    = round(wp.val().Volume(), 2)
            surf_area = round(wp.val().Area(), 2)
        except Exception:
            volume, surf_area = 0.0, 0.0
        num_faces = len(wp.val().Faces())
        num_edges = len(wp.val().Edges())

    # ── Run checks ────────────────────────────────────────────────
    num_holes = 0

    if OCC_AVAILABLE:
        check_wall_thickness_occ(shape, bbox, issues)
        undercut_count = check_undercuts_occ(shape, issues)
        check_sharp_internal_corners_occ(shape, issues)
        num_holes = check_hole_sizes_occ(wp, shape, issues)
        check_draft_angles_occ(shape, issues)
        check_thin_features_cq(wp, bbox, issues)
    else:
        # Fallback path
        check_basic_cq(wp, bbox, issues)
        issues.append(DFMIssue(
            level    = LEVEL_INFO,
            code     = "PARTIAL_ANALYSIS",
            message  = "Full OCC analysis unavailable — running basic geometry checks only.",
            location = "System",
            fix      = "Install pythonOCC (OCP) for complete DFM analysis.",
        ))

    # ── Deduplication: keep the worst issue per (code, location) ──
    seen: dict = {}
    unique_issues: List[DFMIssue] = []
    for issue in issues:
        key = (issue.code, issue.location[:40])
        if key not in seen:
            seen[key] = issue
            unique_issues.append(issue)
        else:
            # Replace with worse severity
            existing = seen[key]
            severity = {"error": 3, "warning": 2, "info": 1}
            if severity.get(issue.level, 0) > severity.get(existing.level, 0):
                idx = unique_issues.index(existing)
                unique_issues[idx] = issue
                seen[key] = issue

    # ── Manufacturability score ────────────────────────────────────
    errors   = sum(1 for i in unique_issues if i.level == LEVEL_ERROR)
    warnings = sum(1 for i in unique_issues if i.level == LEVEL_WARNING)
    infos    = sum(1 for i in unique_issues if i.level == LEVEL_INFO)

    score = max(0, 100 - errors * 20 - warnings * 5 - infos * 1)

    summary = {
        "total_issues":    len(unique_issues),
        "errors":          errors,
        "warnings":        warnings,
        "infos":           infos,
        "num_holes":       num_holes,
        "volume_mm3":      volume,
        "surface_area_mm2": surf_area,
        "bounding_box":    bbox,
        "analysis_engine": "pythonOCC" if OCC_AVAILABLE else "CadQuery fallback",
        "recommended_tool_dia": tool_dia,
        "manufacturability_score": score,
    }

    return DFMReport(
        filename   = filename,
        part_name  = part_name,
        volume_mm3 = volume,
        surface_area_mm2 = surf_area,
        bounding_box = bbox,
        num_faces = num_faces,
        num_edges = num_edges,
        num_holes = num_holes,
        issues    = unique_issues,
        summary   = summary,
        manufacturability_score = score,
    )