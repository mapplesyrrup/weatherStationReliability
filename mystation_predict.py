"""
verify_predictions.py

Backtests the failure-prediction model against your real hurricane case
studies: for each storm, GUESS which stations are about to fail using only
pre-landfall pressure/wind (ISD-Lite), then VERIFY the guess against the
real GHCNd missing-data results your original script (missing_data.py)
already computed.

Pipeline
--------
1. For each of your 5 case studies, pull ISD-Lite hourly data for the
   station network in that hurricane's box, for a short LEAD-UP WINDOW
   immediately before landfall (default: 5 days before `start`).
2. Leave-one-hurricane-out: train the failure model on ISD data from the
   OTHER four storms, so the held-out storm is a genuinely unseen test.
3. Score every ISD station's lead-up window with that model -> a per
   station "guessed as at-risk" flag.
4. Spatially match each ISD station to its nearest GHCNd station (different
   networks, different IDs -- matched by lat/lon within a distance cap).
5. Load the REAL ground truth: your original script's
   `stations_with_missing_data_region.csv` for that storm's landfall window
   (the "after" period in your case_studies dict).
6. Compare: did stations flagged "at risk" actually have high GHCNd
   Missing % during the storm? Report precision, recall, confusion matrix,
   per storm and pooled across all 5.

IMPORTANT -- run order
-----------------------
This script assumes you've already run your original GHCNd script
(missing_data.py) so that, for every case study, this file exists:

    output/<EventName>/<start>_to_<end>/stations_with_missing_data_region.csv

where <start>/<end> are the case's storm-landfall dates (case['start'] /
case['end']), NOT the prev-year control dates. That CSV is your ground
truth. If it's missing for an event, this script skips that event's
verification step (but still shows you the guesses).

Demo mode
---------
Run this file directly with no real data to see the whole pipeline --
leave-one-out split, spatial matching, confusion matrix -- exercised on
synthetic data end to end:

    python verify_predictions.py --demo

Real run
--------
    python verify_predictions.py

which will download real ISD-Lite data (needs `requests`, internet access,
and isd_loader.py in the same folder) and read your real GHCNd outputs.
"""

import argparse
import os
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd

from station_failure_prediction import (
    engineer_features, build_labels, train_model, score as score_rows,
    FEATURE_COLS_TEMPLATE,
)

# ----------------------------------------------------------------------
# CASE STUDIES -- copied from your missing_data.py so both scripts agree
# ----------------------------------------------------------------------

CASE_STUDIES = [
    {"name": "Sandy",
     "lat_min": 39.5, "lat_max": 42.5, "lon_min": -75.5, "lon_max": -72.0,
     "start": "2012-10-25", "end": "2012-11-05"},
    {"name": "Ian",
     "lat_min": 26.0, "lat_max": 34.0, "lon_min": -84.0, "lon_max": -78.5,
     "start": "2022-09-25", "end": "2022-10-05"},
    {"name": "Harvey",
     "lat_min": 27.0, "lat_max": 31.5, "lon_min": -96.5, "lon_max": -91.0,
     "start": "2017-08-23", "end": "2017-09-05"},
    {"name": "Rita",
     "lat_min": 26.0, "lat_max": 33.5, "lon_min": -98.5, "lon_max": -87.0,
     "start": "2005-09-18", "end": "2005-09-26"},
    {"name": "Michael",
     "lat_min": 29.0, "lat_max": 33.0, "lon_min": -86.5, "lon_max": -83.0,
     "start": "2018-10-07", "end": "2018-10-12"},
]

LEAD_DAYS = 5           # how many days before landfall to use for the guess
GHCND_FAIL_THRESHOLD = 50.0   # Missing % that counts as "actually failed" (matches your compute_network_metrics thr=50.0)
MATCH_RADIUS_KM = 25.0        # max distance to consider an ISD station "the same place" as a GHCNd station


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = radians(lat1), radians(lon1), radians(lat2), radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


# ----------------------------------------------------------------------
# STEP 1-2: pull ISD lead-up data + build labeled features per storm
# ----------------------------------------------------------------------

def build_isd_dataset_for_case(case: dict, lead_days: int = LEAD_DAYS, verbose: bool = True) -> pd.DataFrame:
    """
    Real-data version: downloads ISD-Lite for stations in the case's box,
    for [start - lead_days, end], engineers features, and labels rows using
    the station's OWN ISD reporting gaps (is_missing) as the training signal
    (this is what the model learns from -- separate from the GHCNd
    verification data used later).
    """
    from isd_loader import get_stations_in_box, load_isd_lite_region

    start_date = datetime.strptime(case["start"], "%Y-%m-%d")
    end_date = datetime.strptime(case["end"], "%Y-%m-%d")
    lead_start = start_date - timedelta(days=lead_days)

    if verbose:
        print(f"  [{case['name']}] fetching ISD stations in box...")
    stations = get_stations_in_box(case["lat_min"], case["lat_max"],
                                    case["lon_min"], case["lon_max"])

    if verbose:
        print(f"  [{case['name']}] downloading ISD-Lite hourly data "
              f"{lead_start.date()} -> {end_date.date()} ({len(stations)} candidate stations)...")
    years = sorted({lead_start.year, end_date.year})
    df = load_isd_lite_region(stations, years=years,
                               start=lead_start.strftime("%Y-%m-%d"),
                               end=case["end"], verbose=verbose)

    feats = engineer_features(df)
    labeled = build_labels(feats, lead_hours=lead_days * 24)
    labeled["event"] = case["name"]
    labeled["landfall_start"] = start_date
    return labeled


def build_synthetic_isd_dataset_for_case(case: dict, seed: int, lead_days: int = LEAD_DAYS) -> pd.DataFrame:
    """
    Demo-mode stand-in for build_isd_dataset_for_case: generates synthetic
    hourly station data for this case's region, WITHOUT hitting the network,
    so the rest of the pipeline (leave-one-out training, matching, confusion
    matrix) can be exercised end to end.
    """
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp(case["start"])
    end_date = pd.Timestamp(case["end"])
    lead_start = start_date - pd.Timedelta(days=lead_days)
    timestamps = pd.date_range(lead_start, end_date, freq="h")
    n_hours = len(timestamps)

    n_stations = 8
    lats = rng.uniform(case["lat_min"], case["lat_max"], n_stations)
    lons = rng.uniform(case["lon_min"], case["lon_max"], n_stations)
    # roughly half the stations "fail" during this synthetic storm
    fails = rng.choice([True, False], size=n_stations, p=[0.5, 0.5])

    storm_center = int(n_hours * 0.7)
    base_pressure = 1015 - 35 * np.exp(-((np.arange(n_hours) - storm_center) ** 2) / (2 * 15 ** 2))

    rows = []
    for i in range(n_stations):
        stn_id = f"{case['name']}_SYN_{i:02d}"
        pressure = base_pressure + rng.normal(0, 1.0, n_hours)
        wind = np.clip(20 + 50 * np.exp(-((np.arange(n_hours) - storm_center) ** 2) / (2 * 12 ** 2))
                        + rng.normal(0, 2.0, n_hours), 0, None)
        is_missing = np.zeros(n_hours, dtype=bool)

        if fails[i]:
            fail_hour = storm_center + rng.integers(-5, 15)
            fail_hour = max(10, min(fail_hour, n_hours - 1))
            degrade_start = max(0, fail_hour - 15)
            stuck_val = pressure[degrade_start]
            pressure[degrade_start:fail_hour] = stuck_val + rng.normal(0, 0.05, fail_hour - degrade_start)
            wind[degrade_start:fail_hour] += rng.normal(0, 15, fail_hour - degrade_start)
            is_missing[fail_hour:] = True
            pressure[fail_hour:] = np.nan
            wind[fail_hour:] = np.nan

        for h, ts in enumerate(timestamps):
            rows.append({
                "station_id": stn_id, "timestamp": ts,
                "pressure": pressure[h], "wind_speed": wind[h],
                "wind_dir": rng.uniform(0, 360),
                "lat": lats[i], "lon": lons[i],
                "is_missing": is_missing[h],
            })

    df = pd.DataFrame(rows)
    feats = engineer_features(df)
    labeled = build_labels(feats, lead_hours=lead_days * 24)
    labeled["event"] = case["name"]
    labeled["landfall_start"] = start_date
    # stash true fail flag + coords for building synthetic ground truth later
    labeled = labeled.merge(
        pd.DataFrame({"station_id": [f"{case['name']}_SYN_{i:02d}" for i in range(n_stations)],
                       "lat": lats, "lon": lons, "true_failed": fails}),
        on="station_id", how="left", suffixes=("", "_meta")
    )
    return labeled


# ----------------------------------------------------------------------
# STEP 3: leave-one-out train + guess
# ----------------------------------------------------------------------

def guess_at_risk_stations(all_case_data: dict, held_out_event: str,
                            risk_threshold: float = 0.5) -> pd.DataFrame:
    """
    Trains on every event EXCEPT held_out_event, then scores held_out_event's
    lead-up window (rows before landfall_start). Returns one row per station
    in the held-out event with its max risk score and a guessed_at_risk flag.
    """
    train_frames = [df for name, df in all_case_data.items() if name != held_out_event]
    train_data = pd.concat(train_frames, ignore_index=True)
    trained = train_model(train_data, kind="gradient_boosting")

    held_out = all_case_data[held_out_event]
    lead_up = held_out[held_out["timestamp"] < held_out["landfall_start"]]
    scored = score_rows(lead_up, trained)

    per_station = (
        scored.dropna(subset=["risk_score"])
        .groupby("station_id")
        .agg(max_risk_score=("risk_score", "max"),
             lat=("lat", "first"), lon=("lon", "first"))
        .reset_index()
    )
    per_station["guessed_at_risk"] = per_station["max_risk_score"] > risk_threshold
    per_station["event"] = held_out_event
    return per_station


# ----------------------------------------------------------------------
# STEP 4-5: match ISD guesses to GHCNd ground truth
# ----------------------------------------------------------------------

def load_ghcnd_ground_truth(case: dict) -> pd.DataFrame:
    """
    Reads the REAL output your missing_data.py script already produced for
    this storm's landfall window:
        output/<Event>/<start>_to_<end>/stations_with_missing_data_region.csv
    Returns None if the file doesn't exist yet (run missing_data.py first).
    """
    path = os.path.join("output", case["name"], f"{case['start']}_to_{case['end']}",
                         "stations_with_missing_data_region.csv")
    if not os.path.exists(path):
        return None
    gt = pd.read_csv(path)
    gt["actually_failed"] = gt["Missing %"] >= GHCND_FAIL_THRESHOLD
    return gt


def make_synthetic_ground_truth(guesses: pd.DataFrame, all_case_data: dict, event: str) -> pd.DataFrame:
    """Demo-mode stand-in: builds a GHCNd-shaped ground truth table directly
    from the synthetic true_failed flags stashed in build_synthetic_isd_dataset_for_case,
    with its own (slightly offset) lat/lon to simulate a different network."""
    held_out = all_case_data[event]
    meta = held_out[["station_id", "lat", "lon", "true_failed"]].drop_duplicates("station_id")
    rng = np.random.default_rng(abs(hash(event)) % (2**32))
    gt_rows = []
    for _, r in meta.iterrows():
        gt_rows.append({
            "ID": r.station_id.replace("SYN", "GHCND"),
            "LAT": r.lat + rng.normal(0, 0.02),   # small offset = "nearby but different station"
            "LON": r.lon + rng.normal(0, 0.02),
            "Missing %": 80.0 if r.true_failed else rng.uniform(0, 15),
        })
    gt = pd.DataFrame(gt_rows)
    gt["actually_failed"] = gt["Missing %"] >= GHCND_FAIL_THRESHOLD
    return gt


def match_and_compare(guesses: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    For each ISD station with a guess, find the nearest GHCNd station within
    MATCH_RADIUS_KM and attach its actually_failed outcome. Unmatched ISD
    stations are dropped (can't verify them without a nearby GHCNd station).
    """
    matched_rows = []
    for _, g in guesses.iterrows():
        dists = ground_truth.apply(
            lambda r: haversine_km(g.lat, g.lon, r.LAT, r.LON), axis=1
        )
        best_idx = dists.idxmin()
        if dists[best_idx] <= MATCH_RADIUS_KM:
            gt_row = ground_truth.loc[best_idx]
            matched_rows.append({
                "event": g.event,
                "isd_station": g.station_id,
                "matched_ghcnd_station": gt_row["ID"],
                "distance_km": dists[best_idx],
                "guessed_at_risk": g.guessed_at_risk,
                "max_risk_score": g.max_risk_score,
                "actual_missing_pct": gt_row["Missing %"],
                "actually_failed": gt_row["actually_failed"],
            })
    return pd.DataFrame(matched_rows)


# ----------------------------------------------------------------------
# STEP 6: confusion matrix / metrics
# ----------------------------------------------------------------------

def summarize(comparison: pd.DataFrame, label: str) -> dict:
    if comparison.empty:
        print(f"  [{label}] No matched stations -- nothing to summarize.")
        return {}

    tp = ((comparison.guessed_at_risk) & (comparison.actually_failed)).sum()
    fp = ((comparison.guessed_at_risk) & (~comparison.actually_failed)).sum()
    fn = ((~comparison.guessed_at_risk) & (comparison.actually_failed)).sum()
    tn = ((~comparison.guessed_at_risk) & (~comparison.actually_failed)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    accuracy = (tp + tn) / len(comparison)

    print(f"  [{label}] n={len(comparison)}  TP={tp} FP={fp} FN={fn} TN={tn}  "
          f"precision={precision:.2f}  recall={recall:.2f}  accuracy={accuracy:.2f}")

    return {"label": label, "n": len(comparison), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "accuracy": accuracy}


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def run(demo: bool = False, risk_threshold: float = 0.5):
    os.makedirs("outputs", exist_ok=True)
    all_case_data = {}

    print("=== Step 1-2: building ISD lead-up datasets per storm ===")
    for i, case in enumerate(CASE_STUDIES):
        print(f"[{case['name']}]")
        if demo:
            all_case_data[case["name"]] = build_synthetic_isd_dataset_for_case(case, seed=100 + i)
        else:
            all_case_data[case["name"]] = build_isd_dataset_for_case(case)

    print("\n=== Step 3-6: leave-one-out guess + verify, per storm ===")
    all_comparisons = []
    all_metrics = []
    for case in CASE_STUDIES:
        name = case["name"]
        print(f"\n--- Held-out storm: {name} ---")
        guesses = guess_at_risk_stations(all_case_data, name, risk_threshold=risk_threshold)
        print(f"  Guessed {guesses['guessed_at_risk'].sum()} of {len(guesses)} "
              f"ISD stations as at-risk before landfall.")

        if demo:
            ground_truth = make_synthetic_ground_truth(guesses, all_case_data, name)
        else:
            ground_truth = load_ghcnd_ground_truth(case)
            if ground_truth is None:
                print(f"  [skip] No GHCNd ground truth found for {name}. "
                      f"Run missing_data.py for this event's landfall window first.")
                continue

        comparison = match_and_compare(guesses, ground_truth)
        comparison.to_csv(f"outputs/verify_{name}.csv", index=False)
        all_comparisons.append(comparison)
        metrics = summarize(comparison, label=name)
        if metrics:
            all_metrics.append(metrics)

    if all_comparisons:
        pooled = pd.concat(all_comparisons, ignore_index=True)
        pooled.to_csv("outputs/verify_all_events.csv", index=False)
        print("\n=== Pooled across all storms ===")
        pooled_metrics = summarize(pooled, label="ALL EVENTS")
        if pooled_metrics:
            all_metrics.append(pooled_metrics)

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv("outputs/verify_summary_metrics.csv", index=False)
        print("\nSaved outputs/verify_all_events.csv, outputs/verify_summary_metrics.csv, "
              "and outputs/verify_<Event>.csv per storm.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                         help="Run on synthetic data only, no network access needed.")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Risk score threshold above which a station is 'guessed at risk'.")
    args = parser.parse_args()
    run(demo=args.demo, risk_threshold=args.threshold)