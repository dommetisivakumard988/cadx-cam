"""
CadX Studio — CAM Microservice
Production-ready FastAPI server. All config from environment.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

import sentry_sdk
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from config import settings

# ── Logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level    = getattr(logging, settings.log_level.upper(), logging.INFO),
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cadx.cam")

# ── Sentry ─────────────────────────────────────────────────────────

if settings.sentry_dsn:
    sentry_sdk.init(
        dsn          = settings.sentry_dsn,
        environment  = settings.environment,
        integrations = [
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        traces_sample_rate = 0.2 if settings.environment == "production" else 1.0,
        profiles_sample_rate = 0.1,
    )
    logger.info("Sentry initialised for environment: %s", settings.environment)

# ── App ────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "CadX Studio CAM Microservice",
    version     = "1.0.0",
    description = "Geometry analysis, CAM generation, and fixture design for CadX Studio",
    docs_url    = "/docs" if settings.environment != "production" else None,
    redoc_url   = None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = [settings.next_app_url, "http://localhost:3000"],
    allow_methods  = ["POST", "GET", "OPTIONS"],
    allow_headers  = ["*"],
    max_age        = 600,
)

# ── Request logging middleware ─────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("← %s %s %d", request.method, request.url.path, response.status_code)
    return response

# ── Import domain modules ──────────────────────────────────────────

try:
    from cam_engine        import generate_toolpath
    from post_ace_fanuc    import post_process, estimate_cycle_time
    from dfm_checker       import analyse_step
    from fixture_generator import generate_fixture, FixtureRequest
    from feeds_speeds_api  import router as feeds_router
    app.include_router(feeds_router)
    logger.info("All CAM modules loaded successfully")
except ImportError as e:
    logger.error("Module import failed: %s", e)
    # Don't crash the server — individual endpoints will handle missing deps

# ── Health check ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Real health check — verifies each module is importable and
    that required environment variables are present.
    """
    checks: dict[str, str] = {}

    # Module checks
    for module in ["cadquery", "trimesh", "numpy"]:
        try:
            __import__(module)
            checks[module] = "ok"
        except ImportError:
            checks[module] = "missing"

    try:
        import ocl  # noqa: F401
        checks["opencamlib"] = "ok"
    except ImportError:
        checks["opencamlib"] = "missing (fallback active)"

    try:
        from OCC.Core.BRep import BRep_Tool  # noqa: F401
        checks["pythonocc"] = "ok"
    except ImportError:
        checks["pythonocc"] = "missing (cq fallback active)"

    # Config checks
    checks["supabase_url"]         = "ok" if settings.supabase_url         else "missing"
    checks["supabase_service_key"] = "ok" if settings.supabase_service_key else "missing"
    checks["sentry"]               = "ok" if settings.sentry_dsn            else "disabled"

    overall = "ok" if all(v in ("ok", "disabled", "missing (fallback active)",
                                "missing (cq fallback active)")
                          for v in checks.values()) else "degraded"

    return JSONResponse(
        {"status": overall, "environment": settings.environment, "checks": checks},
        status_code=200 if overall == "ok" else 207,
    )

# ── /generate ─────────────────────────────────────────────────────

@app.post("/generate")
async def generate(
    file:            UploadFile = File(...),
    material:        str        = Form("en8"),
    operations:      str        = Form("rough,finish,drill"),
    tool_dia_rough:  float      = Form(16.0),
    tool_dia_finish: float      = Form(10.0),
    machine_max_rpm: int        = Form(6000),
    prog_no:         int        = Form(1),
):
    filename = file.filename or "upload.step"
    if not filename.lower().endswith((".step", ".stp")):
        raise HTTPException(400, "Only STEP (.step/.stp) files are supported")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    content   = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ops_list = [o.strip() for o in operations.split(",") if o.strip()]
        cl_data  = generate_toolpath(
            step_path       = tmp_path,
            material        = material,
            operations_req  = ops_list,
            tool_dia_rough  = tool_dia_rough,
            tool_dia_finish = tool_dia_finish,
            machine_max_rpm = machine_max_rpm,
        )
        gcode, post_warns = post_process(cl_data, prog_no=prog_no)
        cycle             = estimate_cycle_time(cl_data)

        return JSONResponse({
            "success":     True,
            "gcode":       gcode,
            "gcode_lines": len(gcode.split("\n")),
            "part_name":   cl_data.part_name,
            "material":    cl_data.material,
            "bounding_box": cl_data.bounding_box,
            "operations":  [
                {
                    "name":         op.name,
                    "type":         op.op_type,
                    "tool_no":      op.tool_no,
                    "tool_dia":     op.tool_dia,
                    "rpm":          op.cutting.spindle_rpm,
                    "feed":         op.cutting.feed_mmmin,
                    "depth_of_cut": op.cutting.depth_of_cut,
                    "points":       len(op.cl_points),
                    "drills":       len(op.drills),
                    "time_min":     cycle.get(op.name, 0),
                }
                for op in cl_data.operations
            ],
            "cycle_time":  cycle,
            "warnings":    cl_data.warnings + post_warns,
            "machine":     "ACE JT-40 (Fanuc 0i-MF)",
        })
    except Exception as e:
        logger.exception("CAM generation failed for %s", filename)
        sentry_sdk.capture_exception(e)
        raise HTTPException(500, f"CAM generation failed: {e}")
    finally:
        os.unlink(tmp_path)


# ── /dfm/check ────────────────────────────────────────────────────

@app.post("/dfm/check")
async def dfm_check(
    file:     UploadFile = File(...),
    tool_dia: float      = Form(8.0),
):
    filename = file.filename or "upload.step"
    if not filename.lower().endswith((".step", ".stp")):
        raise HTTPException(400, "Only STEP files accepted for DFM analysis")

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        report = analyse_step(tmp_path, tool_dia=tool_dia)
        return JSONResponse(report.to_dict())
    except Exception as e:
        logger.exception("DFM analysis failed for %s", filename)
        sentry_sdk.capture_exception(e)
        raise HTTPException(500, f"DFM analysis failed: {e}")
    finally:
        os.unlink(tmp_path)


# ── /fixture/generate ─────────────────────────────────────────────

@app.post("/fixture/generate")
async def fixture_generate(
    file:                 UploadFile = File(...),
    jaw_length_mm:        float      = Form(...),
    jaw_width_mm:         float      = Form(...),
    jaw_height_mm:        float      = Form(...),
    pocket_depth_mm:      float      = Form(...),
    pocket_clearance_mm:  float      = Form(...),
    pocket_floor_mm:      float      = Form(...),
    add_mounting_holes:   bool       = Form(True),
    bolt_dia_mm:          float      = Form(0.0),
    bolt_pcd_mm:          float      = Form(0.0),
    counterbore:          bool       = Form(True),
    jaw_material:         str        = Form("aluminium_6061"),
    output_format:        str        = Form("step"),
):
    filename = file.filename or "part.step"
    if not filename.lower().endswith((".step", ".stp")):
        raise HTTPException(400, "Only STEP files supported for fixture generation")

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        part_tmp = tmp.name

    try:
        req    = FixtureRequest(
            part_step_path      = part_tmp,
            jaw_length_mm       = jaw_length_mm,
            jaw_width_mm        = jaw_width_mm,
            jaw_height_mm       = jaw_height_mm,
            pocket_depth_mm     = pocket_depth_mm,
            pocket_clearance_mm = pocket_clearance_mm,
            pocket_floor_mm     = pocket_floor_mm,
            add_mounting_holes  = add_mounting_holes,
            bolt_dia_mm         = bolt_dia_mm,
            bolt_pcd_mm         = bolt_pcd_mm,
            counterbore         = counterbore,
            jaw_material        = jaw_material,
            output_format       = output_format,
        )
        result = generate_fixture(req)

        if not result.success:
            raise HTTPException(500, f"Fixture generation failed: {result.warnings}")

        return FileResponse(
            path       = result.output_path,
            filename   = f"{result.jaw_name}.{output_format}",
            media_type = "application/octet-stream",
            headers    = {
                "X-Jaw-Dims":    json.dumps(result.jaw_dims),
                "X-Pocket-Dims": json.dumps(result.pocket_dims),
                "X-Num-Bolts":   str(result.num_bolts),
                "X-Warnings":    json.dumps(result.warnings),
                "X-Part-Name":   result.part_name,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Fixture generation failed for %s", filename)
        sentry_sdk.capture_exception(e)
        raise HTTPException(500, f"Fixture generation failed: {e}")
    finally:
        os.unlink(part_tmp)


# ── Entry point (for local dev) ────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "cam_server:app",
        host    = "0.0.0.0",
        port    = settings.port,
        reload  = settings.environment == "development",
        workers = 1 if settings.environment == "development" else settings.workers,
        log_level = settings.log_level,
    )