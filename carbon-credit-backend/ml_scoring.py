import ee
import geemap
from datetime import datetime
from dateutil.relativedelta import relativedelta

ee.Initialize(project='carbon-credit-mvp')


def run_ml_scoring(coordinates, area_hectares, stage1_output, project_year):
    """
    Main Stage 2 function. Called by Stage 3 backend after Stage 1 completes.

    Six signals weighted into one confidence score:
    1. Tree cover gain  (RF + Dynamic World blend)   — 25%
    2. NDVI dry growth  (dry season delta)           — 20%
    3. Spatial uniformity (std dev check)            — 15%
    4. EVI canopy structure (tree vs crop layers)    — 15%
    5. NDWI irrigation flag (dry season water)       — 10%
    6. Monotonic growth (from Stage 1)               — 15%

    SWIR (B11, B12) is inside the RF classifier features — its influence
    is real but indirect. It discriminates tree canopy from crop rows at
    pixel level, and the output feeds into the 25% tree cover signal.
    """
    geometry        = ee.Geometry.Polygon([coordinates])
    stage1_credits  = stage1_output['stage1_credits']
    dry_season_ndvi = stage1_output['dry_season_ndvi']
    ndvi_baseline   = stage1_output['ndvi_baseline']
    dry_season_evi  = stage1_output.get('dry_season_evi',  0.3)
    dry_season_ndwi = stage1_output.get('dry_season_ndwi', -0.2)
    monotonic_score = stage1_output.get('monotonic_score', 100.0)

    # ── STEP 1: BASELINE LAND COVER (WorldCover 2021) ────────────────────────
    worldcover   = ee.ImageCollection('ESA/WorldCover/v200').first()
    parcel_cover = worldcover.clip(geometry)

    total_area = ee.Image.pixelArea().reduceRegion(
        reducer    = ee.Reducer.sum(),
        geometry   = geometry,
        scale      = 10,
        bestEffort = True
    ).getInfo()['area']

    tree_area_baseline = (parcel_cover.eq(10)
        .multiply(ee.Image.pixelArea())
        .reduceRegion(
            reducer    = ee.Reducer.sum(),
            geometry   = geometry,
            scale      = 10,
            bestEffort = True
        ).getInfo()['Map'])

    baseline_tree_pct = (tree_area_baseline / total_area) * 100
    print(f"Baseline tree cover (WorldCover 2021): {baseline_tree_pct:.1f}%")

    # ── STEP 2: CURRENT IMAGE — dynamic date from Stage 1 output ─────────────
    # Use the actual last checkpoint image date from Stage 1.
    # Build a 3-month window around it to find the best available image.
    # This replaces any hardcoded date.
    last_hash  = stage1_output['image_hashes'][-1]
    last_date  = datetime.strptime(last_hash['image_date'], '%Y-%m-%d')
    date_start = (last_date - relativedelta(months=1)).strftime('%Y-%m-%d')
    date_end   = (last_date + relativedelta(months=2)).strftime('%Y-%m-%d')

    print(f"Current image window: {date_start} to {date_end}")

    current_image = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
        .first()
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    )

    ndvi_band = current_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi_band = current_image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # EVI with correct scaling — Sentinel-2 SR values are x10000 in GEE
    # Without dividing by 10000, EVI produces impossible values like 2.27, 3.06
    evi_band = current_image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))',
        {
            'NIR'  : current_image.select('B8').divide(10000),
            'RED'  : current_image.select('B4').divide(10000),
            'BLUE' : current_image.select('B2').divide(10000)
        }
    ).rename('EVI')

    current_full = current_image.addBands(ndvi_band).addBands(evi_band).addBands(ndwi_band)

    # ── STEP 3: REGIONAL RF TRAINING ─────────────────────────────────────────
    # Train on 50km buffer — not just the project parcel.
    # This fixes WorldCover's young plantation mislabeling problem:
    # the RF learns tree spectral signatures from mature regional forests,
    # then correctly identifies even young plantation pixels in the parcel.
    training_region = geometry.buffer(50000)

    training_points = worldcover.stratifiedSample(
        numPoints  = 200,
        classBand  = 'Map',
        region     = training_region,
        scale      = 10,
        seed       = 42,
        geometries = True
    )

    regional_image = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(training_region)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))
        .median()   # median composite removes cloud/shadow noise from training
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    )

    regional_ndvi = regional_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    regional_ndwi = regional_image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # EVI scaling fix applied to regional training image as well
    regional_evi = regional_image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))',
        {
            'NIR'  : regional_image.select('B8').divide(10000),
            'RED'  : regional_image.select('B4').divide(10000),
            'BLUE' : regional_image.select('B2').divide(10000)
        }
    ).rename('EVI')

    regional_full = (regional_image
        .addBands(regional_ndvi)
        .addBands(regional_evi)
        .addBands(regional_ndwi)
    )

    training_data = regional_full.sampleRegions(
        collection = training_points,
        properties = ['Map'],
        scale      = 10
    )

    # ── STEP 4: TRAIN AND CLASSIFY ────────────────────────────────────────────
    # SWIR (B11, B12) distinguishes tree canopy from crop rows even when
    # NDVI is similar — woody biomass has distinct SWIR signature.
    # EVI and NDWI add canopy structure and water signals to classification.
    classifier = (ee.Classifier.smileRandomForest(numberOfTrees=100)
        .train(
            features        = training_data,
            classProperty   = 'Map',
            inputProperties = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI']
        )
    )

    classified = current_full.classify(classifier)

    # ── STEP 5: CURRENT TREE COVER % ─────────────────────────────────────────
    current_tree_area = (classified.eq(10)
        .multiply(ee.Image.pixelArea())
        .reduceRegion(
            reducer    = ee.Reducer.sum(),
            geometry   = geometry,
            scale      = 10,
            bestEffort = True
        ).getInfo()['classification'])

    rf_tree_pct = (current_tree_area / total_area) * 100

    # ── STEP 6: DYNAMIC WORLD CROSS-VALIDATION ────────────────────────────────
    # Dynamic World is updated continuously — not frozen at 2021 like WorldCover.
    # Blended 60% RF + 40% DW: RF weighted higher because SWIR is included
    # in RF but not exposed in Dynamic World's probability output.
    dw_collection = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterBounds(geometry)
        .filterDate(date_start, date_end)
    )

    dw_trees_mean = (dw_collection
        .select('trees')
        .mean()
        .reduceRegion(
            reducer    = ee.Reducer.mean(),
            geometry   = geometry,
            scale      = 10,
            bestEffort = True
        ).getInfo()
    )

    # Null guard — Dynamic World returns None for cloud-heavy or sparse areas
    dw_tree_pct      = (dw_trees_mean.get('trees') or 0) * 100
    current_tree_pct = (rf_tree_pct * 0.60) + (dw_tree_pct * 0.40)
    tree_cover_gain  = current_tree_pct - baseline_tree_pct

    print(f"RF tree cover      : {rf_tree_pct:.1f}%")
    print(f"Dynamic World      : {dw_tree_pct:.1f}%")
    print(f"Blended tree cover : {current_tree_pct:.1f}%")
    print(f"Tree gain          : +{tree_cover_gain:.1f}%")

    # ── STEP 7: SPATIAL UNIFORMITY ────────────────────────────────────────────
    # Low NDVI std dev = vegetation evenly spread = genuine plantation
    # High std dev = patchy = possible edge-only planting fraud
    ndvi_std = ndvi_band.reduceRegion(
        reducer    = ee.Reducer.stdDev(),
        geometry   = geometry,
        scale      = 10,
        bestEffort = True
    ).getInfo()['NDVI']

    uniformity_score = max(0, min(100, (0.25 - ndvi_std) / (0.25 - 0.10) * 100))

    # ── STEP 8: EVI CANOPY SCORE ──────────────────────────────────────────────
    # After the scaling fix, realistic dry season EVI for plantation: 0.15-0.45
    # Trees (multi-layer canopy): EVI > 0.25  Crops (single-layer): EVI < 0.20
    # Normalise: 0.0 → score 0,  0.40+ → score 100
    evi_score = max(0, min(100, (dry_season_evi / 0.40) * 100))

    # ── STEP 9: NDWI IRRIGATION SCORE ────────────────────────────────────────
    # Natural trees dry season: NDWI -0.4 to -0.1 (low canopy water)
    # Irrigated crops dry season: NDWI -0.1 to +0.2 (high canopy water)
    # Higher NDWI = more suspicious = lower score
    ndwi_score = max(0, min(100, (-dry_season_ndwi + 0.1) / 0.4 * 100))

    # ── STEP 10: AGE-AWARE CONFIDENCE SCORING ────────────────────────────────
    # A 2-year plantation cannot be held to the same standard as a 10-year forest.
    # Ceilings define what "100 score" means for each metric at each project age.
    # project_year is derived from monitoring_periods count in pipeline.py —
    # never hardcoded or passed from a route handler.
    if project_year <= 2:
        tree_gain_ceiling   = 10
        ndvi_growth_ceiling = 0.08
        pass_threshold      = 40
    elif project_year <= 5:
        tree_gain_ceiling   = 20
        ndvi_growth_ceiling = 0.15
        pass_threshold      = 55
    else:
        tree_gain_ceiling   = 40
        ndvi_growth_ceiling = 0.25
        pass_threshold      = 70

    tree_gain_score   = max(0, min(100, (tree_cover_gain / tree_gain_ceiling) * 100))
    ndvi_growth       = dry_season_ndvi - ndvi_baseline
    ndvi_growth_score = max(0, min(100, (ndvi_growth / ndvi_growth_ceiling) * 100))

    confidence_score = (
        tree_gain_score   * 0.25 +
        ndvi_growth_score * 0.20 +
        uniformity_score  * 0.15 +
        evi_score         * 0.15 +
        ndwi_score        * 0.10 +
        monotonic_score   * 0.15
    )

    decision = "PASS" if confidence_score >= pass_threshold else "FAIL"

    # ── STEP 11: CREDITS ──────────────────────────────────────────────────────
    # tree_cover_fraction is the ML adjustment factor applied to Stage 1 credits.
    # It represents what fraction of the declared area is confirmed as tree cover
    # by the RF classifier + Dynamic World ensemble.
    # Applied uniformly across all years — conservative and auditable.
    tree_cover_fraction = current_tree_pct / 100
    adjusted_credits    = int(stage1_credits * tree_cover_fraction) if decision == "PASS" else 0
    buffer_credits      = int(adjusted_credits * 0.15)   # held on-chain, released next verification
    active_credits      = adjusted_credits - buffer_credits  # minted to wallet immediately

    # ── STEP 12: PER-YEAR ML-ADJUSTED CREDIT BREAKDOWN ───────────────────────
    # Apply the same tree_cover_fraction to each year's Stage 1 credits.
    # This gives the auditable per-year breakdown that clients and regulators
    # need to see — not just a total lump sum.
    adjusted_periods = []
    if decision == "PASS":
        for period in stage1_output.get('monitoring_periods', []):
            year_raw       = period['credits']
            year_adjusted  = int(year_raw * tree_cover_fraction)
            year_buffer    = int(year_adjusted * 0.15)
            year_active    = year_adjusted - year_buffer
            adjusted_periods.append({
                "label"           : period['label'],
                "period"          : period.get('period', ''),
                "stage"           : period['stage'],
                "rate"            : period['rate'],
                "start_ndvi"      : period.get('start_ndvi', period.get('ndvi', 0)),
                "end_ndvi"        : period.get('end_ndvi', 0),
                "raw_credits"     : year_raw,
                "adjusted_credits": year_adjusted,
                "buffer_credits"  : year_buffer,
                "active_credits"  : year_active
            })

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    print(f"\n── ML Scoring Results ───────────────────────────────────────────")
    print(f"Project Year      : {project_year}")
    print(f"Pass Threshold    : {pass_threshold}/100")
    print(f"─────────────────────────────────────────────────────────────────")
    print(f"Tree gain score   : {tree_gain_score:.1f}/100   (weight 25%)")
    print(f"NDVI growth score : {ndvi_growth_score:.1f}/100   (weight 20%)")
    print(f"Uniformity score  : {uniformity_score:.1f}/100   (weight 15%)")
    print(f"EVI canopy score  : {evi_score:.1f}/100   (weight 15%)")
    print(f"NDWI irrig score  : {ndwi_score:.1f}/100   (weight 10%)")
    print(f"Monotonic score   : {monotonic_score:.1f}/100   (weight 15%)")
    print(f"─────────────────────────────────────────────────────────────────")
    print(f"Confidence score  : {confidence_score:.1f}/100")
    print(f"Decision          : {decision}")
    print(f"\n── Per-Year Credit Release (ML-Adjusted) ─────────────────────────")
    print(f"Tree cover fraction (ML): {tree_cover_fraction:.1%}")
    if decision == "PASS":
        for p in adjusted_periods:
            print(f"  {p['label']} ({p['period']}): "
                  f"raw={p['raw_credits']}  adjusted={p['adjusted_credits']}  "
                  f"buffer={p['buffer_credits']}  active={p['active_credits']}  "
                  f"[{p['stage']} @ {p['rate']} tCO₂/ha/yr]")
    print(f"─────────────────────────────────────────────────────────────────")
    print(f"Total adjusted    : {adjusted_credits}")
    print(f"Buffer (15%)      : {buffer_credits}  ← held on-chain")
    print(f"Active credits    : {active_credits}  ← minted to wallet")

    return {
        "decision"             : decision,
        "confidence_score"     : round(confidence_score, 2),
        "pass_threshold"       : pass_threshold,
        "baseline_tree_pct"    : round(baseline_tree_pct, 2),
        "current_tree_pct"     : round(current_tree_pct, 2),
        "tree_cover_gain_pct"  : round(tree_cover_gain, 2),
        "rf_tree_pct"          : round(rf_tree_pct, 2),
        "dw_tree_pct"          : round(dw_tree_pct, 2),
        "ndvi_std_dev"         : round(ndvi_std, 4),
        "uniformity_score"     : round(uniformity_score, 2),
        "tree_gain_score"      : round(tree_gain_score, 2),
        "ndvi_growth_score"    : round(ndvi_growth_score, 2),
        "evi_score"            : round(evi_score, 2),
        "ndwi_score"           : round(ndwi_score, 2),
        "monotonic_score"      : round(monotonic_score, 2),
        "stage1_credits"       : stage1_credits,
        "tree_cover_fraction"  : round(tree_cover_fraction, 4),
        "adjusted_credits"     : adjusted_credits,
        "buffer_credits"       : buffer_credits,
        "active_credits"       : active_credits,
        "project_year"         : project_year,
        "adjusted_periods"     : adjusted_periods    # per-year ML-adjusted breakdown
    }


# ── LOCAL TEST — skipped when Stage 3 imports this file ─────────────────────
if __name__ == '__main__':
    # ── KORAPUT CONFIRMED VALUES ─────────────────────────────────────────────
    # These are the actual Stage 1 output values from the confirmed Koraput run.
    # Dry checkpoints: Baseline(0.304), M12(0.295), M36(0.3094), M48(0.4544)
    # → 3 monitoring periods (Year 1, 2, 3) → total 2400 Stage 1 credits
    #
    # FIX: dry_season_evi was 0.9171 (pre-scaling-fix value, physically impossible)
    # Correct value is 0.2349 — actual M48 EVI after applying .divide(10000) fix
    # With wrong value: evi_score = min(100, 229.3) = 100 — masked real signal
    # With correct value: evi_score = (0.2349/0.40)*100 = 58.7 — accurate
    #
    # FIX: project_year was 4 — Koraput has 3 monitoring periods (3 dry-to-dry gaps)
    # Correct value is 3 → age-aware threshold = pass_threshold=55 (year 3-5 bucket)
    mock_stage1 = {
        "stage1_credits"    : 2400,           # 3 years × 800 credits (seedling @ 2 tCO₂/ha/yr)
        "dry_season_ndvi"   : 0.4544,         # M48 dry season NDVI
        "ndvi_baseline"     : 0.304,          # Baseline NDVI (Feb 2018)
        "dry_season_evi"    : 0.2349,         # FIX: M48 EVI after correct scaling (was 0.9171)
        "dry_season_ndwi"   : -0.5331,        # M48 NDWI
        "monotonic_score"   : 100.0,
        "monitoring_periods": [
            # Dry checkpoints: Baseline → M12 → M36 → M48
            # Year 1: Baseline(0.304) → M12(0.295)
            # Year 2: M12(0.295) → M36(0.3094)  ← spans 24 months (M12 to M36)
            # Year 3: M36(0.3094) → M48(0.4544)
            # All seedling stage (NDVI < 0.40) at start of each period
            {"label": "Year 1", "period": "Baseline → M12", "stage": "seedling",
             "rate": 2, "credits": 800, "start_ndvi": 0.304,  "end_ndvi": 0.295},
            {"label": "Year 2", "period": "M12 → M36",      "stage": "seedling",
             "rate": 2, "credits": 800, "start_ndvi": 0.295,  "end_ndvi": 0.3094},
            {"label": "Year 3", "period": "M36 → M48",      "stage": "seedling",
             "rate": 2, "credits": 800, "start_ndvi": 0.3094, "end_ndvi": 0.4544},
        ],
        # Last image hash date — used to build the current image window in Stage 2
        "image_hashes": [{"image_date": "2022-01-05"}]
    }

    test_coords = [
        [82.7100, 18.8200],
        [82.7300, 18.8200],
        [82.7300, 18.8400],
        [82.7100, 18.8400]
    ]

    # FIX: project_year=3, not 4. Koraput has 3 monitoring periods.
    # This puts us in the 3-5 year bucket → pass_threshold=55
    result = run_ml_scoring(test_coords, 400, mock_stage1, project_year=3)
    print("\nFull output:", result)