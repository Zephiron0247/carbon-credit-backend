from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

from dateutil.relativedelta import relativedelta

import sys
import os

# ------------------------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ------------------------------------------------------------------
# ndvi_pipeline.py and ml_scoring.py are located outside
# carbon-credit-backend/ in the project root.
# This makes them importable when uvicorn starts.
# ------------------------------------------------------------------

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from ndvi_pipeline import run_ndvi_pipeline
from ml_scoring import run_ml_scoring


def _build_checkpoints(plantation_date: str) -> List[Tuple[str, str, str]]:
    """
    Build 6-month checkpoint windows from plantation_date.

    Why this exists:
    - Stage 1 needs a checkpoint list.
    - Stage 2 should NOT use DB verification count as ecological age.
    - We derive the analysis window from the plantation date, not from how
      many times the project has been re-verified during debugging.

    Example:
      plantation_date = 2024-01-01
      now             = 2026-05-06
      months_elapsed  = 28
      total_months    = 24
      checkpoints     = Baseline, M6, M12, M18, M24
    """
    plantation_dt = datetime.strptime(plantation_date, "%Y-%m-%d")
    today = datetime.utcnow()

    elapsed = relativedelta(today, plantation_dt)
    months_elapsed = max(0, elapsed.years * 12 + elapsed.months)

    # Only keep fully completed years for checkpoint generation.
    # This matches the existing Stage 1 behavior you've been using.
    total_months = (months_elapsed // 12) * 12

    checkpoints: List[Tuple[str, str, str]] = []

    for month_offset in range(0, total_months + 1, 6):
        start_dt = plantation_dt + relativedelta(months=month_offset)
        end_dt = start_dt + relativedelta(months=3) - relativedelta(days=1)

        label = "Baseline" if month_offset == 0 else f"M{month_offset}"
        checkpoints.append(
            (
                label,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )
        )

    print(
        f"Checkpoints generated : {len(checkpoints)} "
        f"(plantation: {plantation_date} | "
        f"months_elapsed: {months_elapsed} | total_months: {total_months})"
    )

    return checkpoints


def run_full_pipeline(
    coordinates,
    area_hectares,
    plantation_date,
    locked_hashes=None,
    save_chart: bool = False,
):
    """
    Full Stage 1 + Stage 2 pipeline.

    Critical fix:
    - project_year must come from ecological monitoring periods, not from
      the number of database verifications.
    - That keeps the thresholding aligned with the actual plantation age
      being analyzed.

    Returns:
        A merged dict containing Stage 1 fields + Stage 2 fields.
    """
    checkpoints = _build_checkpoints(plantation_date)

    if len(checkpoints) < 2:
        return {"error": "Insufficient checkpoints generated for analysis"}

    stage1_result = run_ndvi_pipeline(
        coordinates=coordinates,
        checkpoints=checkpoints,
        area_hectares=area_hectares,
        save_chart=save_chart,
        locked_hashes=locked_hashes,
    )

    if isinstance(stage1_result, dict) and stage1_result.get("error"):
        return stage1_result

    # Ecological year = number of dry-to-dry monitoring periods, not
    # number of times the project has been re-verified.
    project_year = max(1, len(stage1_result.get("monitoring_periods", [])))

    stage2_result = run_ml_scoring(
        coordinates=coordinates,
        area_hectares=area_hectares,
        stage1_output=stage1_result,
        project_year=project_year,
    )

    if isinstance(stage2_result, dict) and stage2_result.get("error"):
        return stage2_result

    merged: Dict[str, Any] = {}
    merged.update(stage1_result)
    merged.update(stage2_result)

    # Force the correct ecological year into the merged output.
    merged["project_year"] = project_year

    return merged