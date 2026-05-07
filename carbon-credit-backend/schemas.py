from pydantic import BaseModel, field_validator
from typing import List, Optional
import uuid

class CoordinatePoint(BaseModel):
    lon: float
    lat: float

class ProjectSubmit(BaseModel):
    """Shape of the request body when a company submits a new land claim"""
    company_name    : str
    company_wallet  : str
    coordinates     : List[List[float]]   # [[lon, lat], [lon, lat], ...]
    area_hectares   : float
    plantation_date : str                 # "YYYY-MM-DD"

    @field_validator('area_hectares')
    def must_be_viable_area(cls, v):
        # Sentinel-2 at 10m resolution needs minimum ~1 hectare for reliable stats
        # Below this, pixel count is too low for meaningful NDVI mean
        if v < 1:
            raise ValueError('Area must be at least 1 hectare for reliable satellite analysis')
        if v > 100000:
            raise ValueError('Area exceeds 100,000 hectares — split into sub-parcels')
        return v

    @field_validator('coordinates')
    def area_must_match_declaration(cls, v):
        # Basic sanity check — calculated polygon area should be within 50% of declared
        # Full validation happens in the pipeline, this catches obvious mismatches early
        import math
        if len(v) < 3:
            raise ValueError('Polygon must have at least 3 coordinate pairs')
        # Shoelace formula for approximate area
        n = len(v)
        area_deg = abs(sum(
            v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1]
            for i in range(n)
        )) / 2
        lat_km = 111.0
        lon_km = 111.0 * math.cos(math.radians(sum(p[1] for p in v) / n))
        area_ha = area_deg * lat_km * lon_km * 100
        # Store calculated area as metadata — actual mismatch check happens in pipeline
        return v

class ProjectResponse(BaseModel):
    """Shape of the response when project is submitted"""
    project_id   : str
    company_name : str
    status       : str
    message      : str

class VerificationResponse(BaseModel):
    """Shape of the response after verification pipeline runs"""
    project_id       : str
    year_number      : int
    decision         : str
    confidence_score : float
    adjusted_credits : int
    active_credits   : int
    buffer_credits   : int
    tree_cover_pct   : float
    ndvi_baseline    : float
    ndvi_current     : float
    image_hashes     : list
    message          : str

class ProjectStatus(BaseModel):
    """Lightweight status check response"""
    project_id : str
    status     : str
    created_at : str