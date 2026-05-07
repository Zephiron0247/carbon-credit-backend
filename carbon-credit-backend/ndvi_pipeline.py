import ee
import hashlib
import json
import matplotlib.pyplot as plt
from datetime import datetime

ee.Initialize(project='carbon-credit-mvp')

SEQUESTRATION_RATES = {
    'established' : 12,
    'maturing'    : 8,
    'young'       : 5,
    'seedling'    : 2,
    'bare'        : 0
}

def get_stage(ndvi):
    if ndvi >= 0.70:   return 'established'
    elif ndvi >= 0.55: return 'maturing'
    elif ndvi >= 0.40: return 'young'
    elif ndvi >= 0.25: return 'seedling'
    else:              return 'bare'


def get_reference_dry_months(geometry):
    """
    Detects regional phenological dry season using ESTABLISHED natural
    tree cover within 100km of the project site — NOT the project parcel.

    Why reference biome and not the parcel itself:
    Plantation sites are degraded/bare before the project by definition.
    Querying them gives a corrupted dry season signal — bare soil's NDVI
    minimum is in hot summer, not when trees are dormant.
    Nearby established forests experience the same monsoon cycle and give
    the correct phenological signal.

    Works globally: central India Jan-Mar, Chile Jun-Aug, Ethiopia Nov-Jan.
    No country-specific configuration needed.
    """
    buffer_region = geometry.buffer(100000)   # 100km buffer
    worldcover    = ee.ImageCollection('ESA/WorldCover/v200').first()
    tree_mask     = worldcover.eq(10)         # class 10 = tree cover

    # Check if enough established tree cover exists within 100km
    # 50,000,000 m² = 5000 ha — minimum meaningful reference forest
    tree_area = (tree_mask
        .multiply(ee.Image.pixelArea())
        .reduceRegion(
            reducer    = ee.Reducer.sum(),
            geometry   = buffer_region,
            scale      = 100,
            bestEffort = True
        ).getInfo().get('Map', 0) or 0
    )

    if tree_area < 50_000_000:
        print(f"Tree area in 100km: {tree_area/10000:.0f} ha — expanding to 300km")
        buffer_region = geometry.buffer(300000)

    # Query MODIS NDVI on established tree pixels only
    # Date range 2016-2020 avoids post-2020 plantation noise
    modis = (ee.ImageCollection('MODIS/061/MOD13A2')
        .filterBounds(buffer_region)
        .filterDate('2016-01-01', '2020-12-31')
        .select('NDVI')
    )

    monthly_ndvi = []
    for month in range(1, 13):
        result = (modis
            .filter(ee.Filter.calendarRange(month, month, 'month'))
            .mean()
            .updateMask(tree_mask)   # only established tree pixels
            .reduceRegion(
                reducer    = ee.Reducer.mean(),
                geometry   = buffer_region,
                scale      = 500,
                bestEffort = True
            ).getInfo()
        )
        # MODIS NDVI scaled by 0.0001 in GEE — must convert
        ndvi_val = (result.get('NDVI') or 0) * 0.0001
        monthly_ndvi.append((month, ndvi_val))

    sorted_months = sorted(monthly_ndvi, key=lambda x: x[1])
    dry_months    = sorted([m[0] for m in sorted_months[:4]])

    print(f"Reference biome dry months (from established trees nearby): {dry_months}")
    return dry_months


def is_dry_checkpoint(image_date_str, dry_months):
    """Returns True if this checkpoint's image falls in the reference biome dry season."""
    if not dry_months:
        return False
    month = datetime.strptime(image_date_str, "%Y-%m-%d").month
    return month in dry_months


def check_monotonic_growth(dry_ndvi_values):
    """
    Real tree plantations accumulate woody biomass permanently —
    their dry season NDVI floor rises year over year.
    Crops crash after harvest and cannot maintain this pattern.

    Allows small drops (<=0.05) for natural climate variation.
    Penalises larger drops as suspicious.
    Returns: is_consistent (bool), score (0-100), detail string.
    """
    if len(dry_ndvi_values) < 2:
        return True, 100.0, "Insufficient data for monotonic check"

    penalties = 0
    details   = []

    for i in range(1, len(dry_ndvi_values)):
        drop = dry_ndvi_values[i-1] - dry_ndvi_values[i]
        if drop > 0.10:
            penalties += 40
            details.append(f"Severe drop {drop:.3f} at period {i}→{i+1} — likely harvest")
        elif drop > 0.05:
            penalties += 20
            details.append(f"Moderate drop {drop:.3f} at period {i}→{i+1} — suspicious")

    score         = max(0, 100 - penalties)
    is_consistent = score >= 60
    detail_str    = "; ".join(details) if details else "Growth pattern consistent with woody vegetation"

    return is_consistent, score, detail_str


def get_best_image(geometry, start_date, end_date, cloud_threshold=35):
    """
    Pulls the least cloudy Sentinel-2 image for a given time window.
    Returns image, metadata dict, and count of available images.
    """
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    )

    count = collection.size().getInfo()
    if count == 0:
        return None, None, count

    image = collection.first()
    props_check = image.toDictionary(['CLOUDY_PIXEL_PERCENTAGE']).getInfo()

    # Reject single images with high cloud cover — one bad image is worse than no image
    # Multiple images in a collection means GEE already filtered to the clearest one
    # A single image with >20% cloud is unreliable and will corrupt NDVI readings
    if count == 1 and props_check.get('CLOUDY_PIXEL_PERCENTAGE', 0) > 20:
        return None, None, count
    props = image.toDictionary(['system:time_start', 'CLOUDY_PIXEL_PERCENTAGE']).getInfo()

    image_date = datetime.fromtimestamp(
        props['system:time_start'] / 1000
    ).strftime('%Y-%m-%d')

    metadata = {
        "image_date" : image_date,
        "cloud_pct"  : round(props['CLOUDY_PIXEL_PERCENTAGE'], 2),
        "count"      : count
    }

    return image, metadata, count


def compute_indices(image, geometry):
    """
    Computes NDVI, EVI, NDWI for a Sentinel-2 image.

    NDVI and NDWI use normalizedDifference — the band ratio cancels
    the GEE 10000 scale factor automatically. No manual scaling needed.

    EVI uses an explicit formula with constants tuned for 0-1 reflectance.
    Sentinel-2 SR bands in GEE are scaled by 10000, so we divide first.
    Without this correction, EVI produces physically impossible values >1.0
    (e.g. 2.27, 3.06 seen in earlier runs — those were all wrong).

    EVI captures multi-layer canopy structure: trees score higher than crops.
    NDWI captures canopy water: high in dry season = irrigation = suspicious.
    """
    # Ratio indices — 10000 scaling cancels automatically
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # EVI — explicit formula requires 0-1 reflectance, so divide by 10000 first
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))',
        {
            'NIR'  : image.select('B8').divide(10000),
            'RED'  : image.select('B4').divide(10000),
            'BLUE' : image.select('B2').divide(10000)
        }
    ).rename('EVI')

    stats = (ndvi.addBands(evi).addBands(ndwi)
        .reduceRegion(
            reducer    = ee.Reducer.mean(),
            geometry   = geometry,
            scale      = 10,
            bestEffort = True
        ).getInfo()
    )

    ndvi_std_stat = ndvi.reduceRegion(
        reducer    = ee.Reducer.stdDev(),
        geometry   = geometry,
        scale      = 10,
        bestEffort = True
    ).getInfo()

    return {
        'ndvi'     : stats.get('NDVI'),
        'evi'      : stats.get('EVI'),
        'ndwi'     : stats.get('NDWI'),
        'ndvi_std' : ndvi_std_stat.get('NDVI'),
    }, ndvi, evi, ndwi


def generate_image_hash(coordinates, image_date, ndvi_value, cloud_pct, checkpoint_label):
    """
    SHA-256 hash of the exact parameters that produced this NDVI reading.
    Anyone can independently rerun the same GEE query to verify.
    This hash goes on-chain — proves the data was never altered after analysis.
    """
    payload = {
        "coordinates"         : coordinates,
        "image_date"          : image_date,
        "ndvi"                : round(ndvi_value, 6),
        "cloud_pct"           : cloud_pct,
        "checkpoint"          : checkpoint_label,
        "sentinel_collection" : "COPERNICUS/S2_SR_HARMONIZED"
    }
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode()).hexdigest()


def run_ndvi_pipeline(coordinates, checkpoints, area_hectares, save_chart=False, locked_hashes=None):
    """
    Main Stage 1 function. Called by Stage 3 backend via pipeline.py.

    FIX (Point 4 — NDVI reproducibility across surveys):
    locked_hashes parameter added. For checkpoints that were already verified
    in a prior annual survey, Stage 1 uses the STORED values from the database
    instead of re-querying GEE. This guarantees that Year 1 NDVI of 0.304
    (recorded on-chain in 2019) will always read as 0.304 when Year 2 runs in
    2020 — even if GEE's image catalog has been updated. The SHA-256 hash of
    each checkpoint is the immutable proof of what was measured.

    Args:
        coordinates   : list of [lon, lat] pairs forming the polygon
        checkpoints   : list of (label, start_date, end_date) tuples
        area_hectares : float — parcel area
        save_chart    : bool — True only when running locally for debugging
        locked_hashes : list of image_hash dicts from DB (previously verified
                        checkpoints). These are used AS-IS — no GEE call made.

    Returns:
        dict with all values Stage 2 and Stage 3 need
    """
    geometry = ee.Geometry.Polygon([coordinates])

    # Detect regional dry season from established forests nearby — not the parcel
    dry_months = get_reference_dry_months(geometry)

    # Build lookup of locked checkpoints by label
    # These come from the DB — previously verified, stored, immutable
    # -------------------------------------------------------------------------
    # SAFE LOCKED HASH LOOKUP
    # -------------------------------------------------------------------------
    # Converts locked hashes into dictionary lookup.
    # Handles corrupted or unexpected formats safely.
    # -------------------------------------------------------------------------

    locked_lookup = {}

    if locked_hashes:

        for h in locked_hashes:

            # Ensure object is dictionary
            if isinstance(h, dict):

                checkpoint = h.get("checkpoint")

                if checkpoint:
                    locked_lookup[checkpoint] = h

            else:

                print(
                    f"[WARNING] Invalid locked_hash format skipped: {h}"
                )

    ndvi_values  = []
    evi_values   = []
    ndwi_values  = []
    labels       = []
    image_hashes = []
    skipped      = []

    for label, start, end in checkpoints:

        # ── LOCKED CHECKPOINT — use stored DB value, skip GEE ────────────────
        # This is the reproducibility guarantee. A checkpoint locked in Year 1
        # will always produce the same NDVI in Year 2, 3, 4 verifications.
        # The SHA-256 hash already on-chain will continue to match.
        if label in locked_lookup:
            locked   = locked_lookup[label]
            ndvi_val = locked['ndvi']
            evi_val  = locked.get('evi', 0.0)
            ndwi_val = locked.get('ndwi', 0.0)
            is_dry   = locked.get('is_dry', False)
            img_hash = locked['image_hash']
            count    = locked.get('image_count', 0)

            image_hashes.append({
                "checkpoint"  : label,
                "image_date"  : locked['image_date'],
                "cloud_pct"   : locked['cloud_pct'],
                "ndvi"        : ndvi_val,
                "evi"         : evi_val,
                "ndwi"        : ndwi_val,
                "is_dry"      : is_dry,
                "image_hash"  : img_hash,
                "image_count" : count,
                "locked"      : True   # audit flag — came from DB, not re-computed
            })
            ndvi_values.append(ndvi_val)
            evi_values.append(evi_val)
            ndwi_values.append(ndwi_val)
            labels.append(label)
            print(f"{label}: LOCKED (DB) NDVI={ndvi_val}  EVI={evi_val}  NDWI={ndwi_val}  is_dry={is_dry}")
            continue

        # ── LIVE GEE QUERY — new checkpoint not yet in DB ────────────────────
        image, metadata, count = get_best_image(geometry, start, end)

        if image is None:
            print(f"{label}: NO CLEAR IMAGE FOUND — skipping")
            skipped.append(label)
            continue

        image_bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        indices, _, _, _ = compute_indices(image_bands, geometry)

        if indices['ndvi'] is None:
            print(f"{label}: NDVI NULL — skipping")
            skipped.append(label)
            continue

        ndvi_val = round(indices['ndvi'], 4)
        evi_val  = round(indices['evi'],  4) if indices['evi']  is not None else 0.0
        ndwi_val = round(indices['ndwi'], 4) if indices['ndwi'] is not None else 0.0

        # Tag whether this checkpoint falls in the reference biome dry season
        # Stored as audit metadata in the hash — useful for on-chain verification
        is_dry   = is_dry_checkpoint(metadata['image_date'], dry_months)

        img_hash = generate_image_hash(
            coordinates      = coordinates,
            image_date       = metadata['image_date'],
            ndvi_value       = ndvi_val,
            cloud_pct        = metadata['cloud_pct'],
            checkpoint_label = label
        )

        image_hashes.append({
            "checkpoint"  : label,
            "image_date"  : metadata['image_date'],
            "cloud_pct"   : metadata['cloud_pct'],
            "ndvi"        : ndvi_val,
            "evi"         : evi_val,
            "ndwi"        : ndwi_val,
            "is_dry"      : is_dry,
            "image_hash"  : img_hash,
            "image_count" : count,
            "locked"      : False  # fresh from GEE — will be locked after this verification
        })

        ndvi_values.append(ndvi_val)
        evi_values.append(evi_val)
        ndwi_values.append(ndwi_val)
        labels.append(label)

        print(f"{label}: NDVI={ndvi_val}  EVI={evi_val}  NDWI={ndwi_val}  dry={is_dry}  ({count} images)")

    if len(ndvi_values) < 2:
        return {"error": "Insufficient clear images for analysis"}

    # ── DRY CHECKPOINT DETECTION ──────────────────────────────────────────────
    # Primary method: local NDVI minima on the curve.
    # A checkpoint is dry when NDVI is lower than both neighbours —
    # grass disappears in dry season, permanent woody vegetation stays.
    # Reference biome months (MODIS) are stored as informational audit metadata only.
    dry_indices = []
    for i in range(len(ndvi_values)):
        left  = ndvi_values[i-1] if i > 0 else float('inf')
        right = ndvi_values[i+1] if i < len(ndvi_values)-1 else float('inf')
        if ndvi_values[i] <= left and ndvi_values[i] <= right:
            dry_indices.append(i)

    # Always anchor on baseline (index 0) and latest checkpoint
    if 0 not in dry_indices:
        dry_indices.insert(0, 0)
    if len(ndvi_values)-1 not in dry_indices:
        dry_indices.append(len(ndvi_values)-1)

    dry_indices = sorted(set(dry_indices))

    # Informational check — does reference biome agree with local minima?
    print(f"\nDry checkpoints identified : {[labels[i] for i in dry_indices]}")
    print(f"MODIS reference dry months : {dry_months}")
    modis_agreement = all(
        datetime.strptime(image_hashes[i]['image_date'], '%Y-%m-%d').month in dry_months
        for i in dry_indices if i < len(image_hashes)
    )
    print(f"MODIS agreement            : {'YES' if modis_agreement else 'PARTIAL — review checkpoint dates'}")

    dry_ndvi_values = [ndvi_values[i] for i in dry_indices]
    dry_evi_values  = [evi_values[i]  for i in dry_indices]
    dry_ndwi_values = [ndwi_values[i] for i in dry_indices]

    ndvi_baseline   = dry_ndvi_values[0]
    dry_season_ndvi = dry_ndvi_values[-1]
    dry_season_evi  = dry_evi_values[-1]
    dry_season_ndwi = dry_ndwi_values[-1]

    is_monotonic, monotonic_score, monotonic_detail = check_monotonic_growth(dry_ndvi_values)
    print(f"Monotonic check: {monotonic_detail}  (score: {monotonic_score})")

    # ── PER-PERIOD CREDIT CALCULATION ────────────────────────────────────────
    # Each dry-to-dry gap = one verified year of sequestration.
    # Rate uses the NDVI at the START of each period (conservative — IPCC approach).
    # Annual issuance — credits released every year, not at project end.
    total_credits      = 0
    monitoring_periods = []

    print(f"\n── Year-wise Credit Breakdown ───────────────────────────────────────")
    for i in range(1, len(dry_indices)):
        period_start_label = labels[dry_indices[i-1]]
        period_end_label   = labels[dry_indices[i]]
        period_ndvi        = dry_ndvi_values[i-1]   # rate = NDVI at START
        stage              = get_stage(period_ndvi)
        rate               = SEQUESTRATION_RATES[stage]
        credits            = int(area_hectares * rate * 1.0)
        total_credits     += credits

        monitoring_periods.append({
            "label"        : f"Year {i}",
            "period"       : f"{period_start_label} → {period_end_label}",
            "start_ndvi"   : period_ndvi,
            "end_ndvi"     : dry_ndvi_values[i],
            "stage"        : stage,
            "rate"         : rate,
            "credits"      : credits
        })
        print(f"  Year {i} ({period_start_label} → {period_end_label}): "
              f"NDVI {period_ndvi:.4f} → {dry_ndvi_values[i]:.4f}  "
              f"stage={stage}  rate={rate} tCO₂/ha/yr  credits={credits}")

    print(f"\nTotal Stage 1 credits across all years: {total_credits}")

    if save_chart:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(labels, ndvi_values,  marker='o', color='green',  linewidth=2, label='NDVI')
        ax1.plot(labels, evi_values,   marker='s', color='blue',   linewidth=2, label='EVI')
        ax1.plot(labels, ndwi_values,  marker='^', color='purple', linewidth=2, label='NDWI')
        ax1.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7, label='Seedling (0.25)')
        ax1.axhline(y=0.40, color='blue',   linestyle='--', alpha=0.7, label='Young (0.40)')
        ax1.axhline(y=0.55, color='red',    linestyle='--', alpha=0.7, label='Maturing (0.55)')
        ax1.set_title('Vegetation Indices — Koraput, Odisha')
        ax1.set_ylabel('Index Value')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=15)

        years = [p["label"]   for p in monitoring_periods]
        creds = [p["credits"] for p in monitoring_periods]
        ax2.bar(years, creds, color='darkgreen', alpha=0.8)
        ax2.set_title('Credits Per Year (Stage 1 — before ML adjustment)')
        ax2.set_ylabel('Credits (tCO₂)')
        for j, v in enumerate(creds):
            ax2.text(j, v + 10, str(v), ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('ndvi_timeseries.png')
        plt.close()
        print("Chart saved: ndvi_timeseries.png")

    return {
        "ndvi_baseline"      : ndvi_baseline,
        "dry_season_ndvi"    : dry_season_ndvi,
        "dry_season_evi"     : dry_season_evi,
        "dry_season_ndwi"    : dry_season_ndwi,
        "ndvi_values"        : ndvi_values,
        "evi_values"         : evi_values,
        "ndwi_values"        : ndwi_values,
        "labels"             : labels,
        "stage1_credits"     : total_credits,
        "monitoring_periods" : monitoring_periods,
        "image_hashes"       : image_hashes,
        "skipped"            : skipped,
        "dry_months"         : dry_months,
        "dry_indices"        : dry_indices,
        "dry_ndvi_values"    : dry_ndvi_values,
        "monotonic_score"    : monotonic_score,
        "monotonic_detail"   : monotonic_detail,
        "is_monotonic"       : is_monotonic
    }


# ── LOCAL TEST — skipped when Stage 3 imports this file ─────────────────────
if __name__ == '__main__':
    test_coords = [
        [82.7100, 18.8200],
        [82.7300, 18.8200],
        [82.7300, 18.8400],
        [82.7100, 18.8400]
    ]
    test_checkpoints = [
        ("Baseline", "2018-01-01", "2018-03-31"),
        ("M6",       "2018-07-01", "2018-09-30"),
        ("M12",      "2019-01-01", "2019-03-31"),
        ("M18",      "2019-07-01", "2019-09-30"),
        ("M24",      "2020-01-01", "2020-03-31"),
        ("M30",      "2020-07-01", "2020-09-30"),
        ("M36",      "2021-01-01", "2021-03-31"),
        ("M42",      "2021-07-01", "2021-09-30"),
        ("M48",      "2022-01-01", "2022-03-31"),
    ]
    result = run_ndvi_pipeline(test_coords, test_checkpoints, 400, save_chart=True)
    print(f"\n── Stage 1 Complete ──────────────────────────────────────────────")
    print(f"Baseline NDVI    : {result['ndvi_baseline']}")
    print(f"Current NDVI     : {result['dry_season_ndvi']}")
    print(f"Stage 1 Credits  : {result['stage1_credits']}")
    print(f"\nYear-wise breakdown:")
    for p in result['monitoring_periods']:
        print(f"  {p['label']} ({p['period']}): {p['credits']} credits  "
              f"[{p['stage']} @ {p['rate']} tCO₂/ha/yr]  "
              f"NDVI {p['start_ndvi']:.4f} → {p['end_ndvi']:.4f}")