"""
verify_predictions.py

Backtests the failure-prediction model against your real hurricane case
studies: for each storm, GUESS which stations are about to fail using only
pre-landfall pressure/wind (ISD-Lite), then VERIFY the guess against the
real GHCNd missing-data results your original script (caseStudies.py)
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
(caseStudies.py) so that, for every case study, this file exists:

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

from station_predict import (
    engineer_features, build_labels, train_model, score as score_rows,
    FEATURE_COLS_TEMPLATE,
)

# ----------------------------------------------------------------------
# CASE STUDIES -- copied from your caseStudies.py so both scripts agree
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

def _score_lead_up_per_station(case_data: pd.DataFrame, trained) -> pd.DataFrame:
    """Score one event's pre-landfall window and collapse to one row per
    station: its max risk score, plus whether it was EVER labeled
    'about to fail' in that window (station_positive) -- the same
    aggregation used for the real guess, so calibration and application
    are apples-to-apples."""
    lead_up = case_data[case_data["timestamp"] < case_data["landfall_start"]]
    scored = score_rows(lead_up, trained)
    return (
        scored.dropna(subset=["risk_score"])
        .groupby("station_id")
        .agg(max_risk_score=("risk_score", "max"),
             station_positive=("label", "max"),
             lat=("lat", "first"), lon=("lon", "first"))
        .reset_index()
    )


def calibrate_threshold(all_case_data: dict, training_events: list) -> float:
    """
    Picks a risk threshold using ONLY the training storms (no leakage from
    the held-out storm's true outcomes). Does a nested leave-one-out among
    the training storms: for each one, train on the REST, score its own
    lead-up window, and collect (max_risk_score, station_positive) pairs
    at the station level -- the same unit the real guess is made at. Then
    picks the threshold that maximizes F1 across all that pooled,
    out-of-fold evidence.
    """
    from sklearn.metrics import f1_score

    station_rows = []
    for name in training_events:
        inner_train_names = [n for n in training_events if n != name]
        if not inner_train_names:
            continue
        inner_train_data = pd.concat([all_case_data[n] for n in inner_train_names], ignore_index=True)
        inner_model = train_model(inner_train_data, kind="gradient_boosting")
        per_station = _score_lead_up_per_station(all_case_data[name], inner_model)
        station_rows.append(per_station)

    if not station_rows:
        return 0.5  # fallback, shouldn't normally happen with 5 events

    pooled = pd.concat(station_rows, ignore_index=True)
    if pooled["station_positive"].nunique() < 2:
        # no positive/negative contrast to calibrate against -- fall back
        # to flagging the top 20% highest-risk stations
        return pooled["max_risk_score"].quantile(0.80)

    scores = pooled["max_risk_score"].values
    y = pooled["station_positive"].astype(int).values
    candidates = np.unique(scores)
    best_thr, best_f1 = candidates[0], -1.0
    for thr in candidates:
        preds = (scores > thr).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


def guess_at_risk_stations(all_case_data: dict, held_out_event: str,
                            risk_threshold: float = None) -> tuple:
    """
    Trains on every event EXCEPT held_out_event, then scores held_out_event's
    lead-up window (rows before landfall_start). Returns:
      - one row per station in the held-out event with its max risk score
        and a guessed_at_risk flag
      - the threshold that was used
      - a DataFrame of feature importances from this fold's model

    Threshold: if risk_threshold is None (default), it's calibrated by
    calibrate_threshold() using ONLY the training storms via nested
    leave-one-out -- the held-out storm's true outcomes are never touched
    until the final comparison step.
    """
    train_frames = [df for name, df in all_case_data.items() if name != held_out_event]
    train_data = pd.concat(train_frames, ignore_index=True)
    trained = train_model(train_data, kind="gradient_boosting")

    importances = pd.DataFrame({
        "feature": trained.feature_cols,
        "importance": trained.model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances["held_out_event"] = held_out_event

    if risk_threshold is None:
        training_events = [n for n in all_case_data if n != held_out_event]
        risk_threshold = calibrate_threshold(all_case_data, training_events)

    per_station = _score_lead_up_per_station(all_case_data[held_out_event], trained)
    per_station["guessed_at_risk"] = per_station["max_risk_score"] > risk_threshold
    per_station["event"] = held_out_event
    per_station["threshold_used"] = risk_threshold
    return per_station, risk_threshold, importances


# ----------------------------------------------------------------------
# STEP 4-5: match ISD guesses to GHCNd ground truth
# ----------------------------------------------------------------------

def load_ghcnd_ground_truth(case: dict) -> pd.DataFrame:
    """
    Reads the REAL output your caseStudies.py script already produced for
    this storm's landfall window:
        output/<Event>/<start>_to_<end>/stations_with_missing_data_region.csv
    Returns None if the file doesn't exist yet (run caseStudies.py first).
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

    # AUC / average precision use the CONTINUOUS risk score rather than the
    # thresholded guess -- this tells you whether the model has any signal
    # at all, independent of where the threshold happens to be set.
    auc, ap = np.nan, np.nan
    if comparison["actually_failed"].nunique() == 2:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(comparison["actually_failed"], comparison["max_risk_score"])
        ap = average_precision_score(comparison["actually_failed"], comparison["max_risk_score"])

    print(f"  [{label}] n={len(comparison)}  TP={tp} FP={fp} FN={fn} TN={tn}  "
          f"precision={precision:.2f}  recall={recall:.2f}  accuracy={accuracy:.2f}  "
          f"AUC={auc:.3f}  AP={ap:.3f}")

    return {"label": label, "n": len(comparison), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "accuracy": accuracy,
            "auc": auc, "average_precision": ap}


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def run(demo: bool = False, risk_threshold: float = None):
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
    all_importances = []
    for case in CASE_STUDIES:
        name = case["name"]
        print(f"\n--- Held-out storm: {name} ---")
        guesses, threshold_used, importances = guess_at_risk_stations(
            all_case_data, name, risk_threshold=risk_threshold)
        all_importances.append(importances)
        print(f"  Calibrated threshold for this fold: {threshold_used:.4f}")
        print(f"  Guessed {guesses['guessed_at_risk'].sum()} of {len(guesses)} "
              f"ISD stations as at-risk before landfall.")
        print(f"  Top 3 features this fold: "
              f"{', '.join(importances.head(3)['feature'].tolist())}")

        if demo:
            ground_truth = make_synthetic_ground_truth(guesses, all_case_data, name)
        else:
            ground_truth = load_ghcnd_ground_truth(case)
            if ground_truth is None:
                print(f"  [skip] No GHCNd ground truth found for {name}. "
                      f"Run caseStudies.py for this event's landfall window first.")
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

    if all_importances:
        imp_all = pd.concat(all_importances, ignore_index=True)
        imp_all.to_csv("outputs/feature_importances_by_fold.csv", index=False)

        imp_avg = (
            imp_all.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        imp_avg.to_csv("outputs/feature_importances_averaged.csv", index=False)

        print("\n=== Feature importance, averaged across all 5 leave-one-out folds ===")
        for _, row in imp_avg.iterrows():
            bar = "#" * int(row["importance"] * 100)
            print(f"  {row['feature']:<28} {row['importance']:.3f}  {bar}")

    print("\nOutputs saved to outputs/: verify_<Event>.csv, verify_all_events.csv, "
          "verify_summary_metrics.csv, feature_importances_by_fold.csv, "
          "feature_importances_averaged.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                         help="Run on synthetic data only, no network access needed.")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Fixed risk score threshold above which a station is 'guessed "
                              "at risk'. If omitted (default), the threshold is calibrated "
                              "automatically per fold from the training data's own positive "
                              "rate -- no leakage from the held-out storm's true outcomes.")
    args = parser.parse_args()
    run(demo=args.demo, risk_threshold=args.threshold)