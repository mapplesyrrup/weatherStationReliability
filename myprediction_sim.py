"""








# pp

























station_failure_prediction.py

Early-warning model for weather-station failure during extreme precipitation
events (hurricanes), using ONLY a station's own past pressure/wind readings
(plus optional comparison to nearby "neighbor" stations).

Pipeline
--------
1. load_station_data()       -> load hourly station observations
2. engineer_features()       -> rolling stats, stuck-sensor flags, neighbor divergence
3. build_labels()            -> mark "failure onset" and pre-failure lead-time windows
4. train_model()             -> IsolationForest baseline + optional GradientBoosting
5. evaluate()                -> lead-time distribution, false alarm rate, PR curve
6. score_live()               -> apply trained model to a rolling live buffer

Run this file directly to see a full demo on synthetic data:
    python station_failure_prediction.py

To use with your real data, replace load_station_data() with a loader that
reads your ISD / GHCNd hourly pulls into the same schema:

    station_id | timestamp | pressure | wind_speed | wind_dir | lat | lon

Author: (adapt as needed for your project)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import GroupShuffleSplit


# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------

def load_station_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load hourly station observations into a long-format DataFrame with columns:
        station_id, timestamp, pressure, wind_speed, wind_dir, lat, lon

    If `path` is None, generates synthetic demo data for several stations,
    including a few that "fail" (sensor drift -> stuck values -> data loss)
    partway through a simulated storm.

    Real data sources to plug in here:
      - NOAA Integrated Surface Database (ISD) hourly files
      - GHCNd (daily only -- too coarse for this early-warning use case,
        but fine for validating your failure labels against the paper's
        missing_summary.csv)
    """
    if path is not None:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    return _make_synthetic_data()


def _make_synthetic_data(n_hours: int = 24 * 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly data for a handful of stations approaching a storm."""
    rng = np.random.default_rng(seed)
    stations = {
        "STN_A_coastal_fails": {"lat": 40.6, "lon": -73.9, "fails": True, "fail_hour": 170},
        "STN_B_coastal_ok":    {"lat": 40.7, "lon": -74.0, "fails": False, "fail_hour": None},
        "STN_C_inland_fails":  {"lat": 40.9, "lon": -74.3, "fails": True, "fail_hour": 190},
        "STN_D_inland_ok":     {"lat": 41.0, "lon": -74.5, "fails": False, "fail_hour": None},
        "STN_E_coastal_fails": {"lat": 40.5, "lon": -73.8, "fails": True, "fail_hour": 160},
    }

    timestamps = pd.date_range("2012-10-24", periods=n_hours, freq="h")
    # simulated storm: pressure dips to a minimum around hour 175
    storm_center = 175
    base_pressure = 1015 - 40 * np.exp(-((np.arange(n_hours) - storm_center) ** 2) / (2 * 20 ** 2))
    base_wind = 10 + 60 * np.exp(-((np.arange(n_hours) - storm_center) ** 2) / (2 * 15 ** 2))

    rows = []
    for stn_id, meta in stations.items():
        pressure = base_pressure + rng.normal(0, 1.0, n_hours)
        wind = np.clip(base_wind + rng.normal(0, 2.0, n_hours), 0, None)
        wind_dir = (rng.uniform(0, 360, n_hours))

        is_missing = np.zeros(n_hours, dtype=bool)

        if meta["fails"]:
            fh = meta["fail_hour"]
            # Pre-failure degradation window: sensor starts behaving oddly
            # ~15 hours before full data loss (stuck values + erratic jumps).
            degrade_start = max(0, fh - 15)
            # stuck-value stretch (classic pre-failure signature)
            stuck_val = pressure[degrade_start]
            pressure[degrade_start:fh] = stuck_val + rng.normal(0, 0.05, fh - degrade_start)
            # erratic wind spikes right before death
            wind[degrade_start:fh] += rng.normal(0, 15, fh - degrade_start)
            # full failure: missing data from fail_hour onward
            is_missing[fh:] = True
            pressure[fh:] = np.nan
            wind[fh:] = np.nan
            wind_dir[fh:] = np.nan

        for i, ts in enumerate(timestamps):
            rows.append({
                "station_id": stn_id,
                "timestamp": ts,
                "pressure": pressure[i],
                "wind_speed": wind[i],
                "wind_dir": wind_dir[i],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "is_missing": is_missing[i],
            })

    return pd.DataFrame(rows).sort_values(["station_id", "timestamp"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_neighbor_divergence(df: pd.DataFrame, k: int = 3, radius_km: float = 150) -> pd.DataFrame:
    """
    For each station+timestamp, compare its reading to the mean of its k
    nearest neighboring stations (within radius_km) at the SAME timestamp.
    Large divergence suggests a station-specific sensor problem rather than
    a real regional weather event (which would show up at neighbors too).
    """
    coords = df[["station_id", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    coords["lat"] = coords["lat"].astype(float)
    coords["lon"] = coords["lon"].astype(float)
    neighbor_map = {}
    for _, row in coords.iterrows():
        dists = _haversine_km(float(row.lat), float(row.lon), coords.lat.values, coords.lon.values)
        order = np.argsort(dists)
        nearby = coords.station_id.values[order]
        nearby_d = dists[order]
        nearby = nearby[(nearby != row.station_id) & (nearby_d <= radius_km)][:k]
        neighbor_map[row.station_id] = list(nearby)

    out = df.copy()
    any_neighbors = any(len(v) > 0 for v in neighbor_map.values())
    if not any_neighbors:
        out["pressure_neighbor_div"] = 0.0
        out["wind_speed_neighbor_div"] = 0.0
        return out

    for var in ["pressure", "wind_speed"]:
        pivot = out.pivot(index="timestamp", columns="station_id", values=var)
        div_col = f"{var}_neighbor_div"
        out[div_col] = np.nan
        for stn, neighbors in neighbor_map.items():
            if not neighbors:
                continue
            mask = out.station_id == stn
            neighbor_mean = pivot[neighbors].mean(axis=1)
            station_vals = pivot[stn]
            diff = (station_vals - neighbor_mean).reindex(out.loc[mask, "timestamp"]).values
            out.loc[mask, div_col] = diff
    return out


def engineer_features(df: pd.DataFrame, windows=(3, 6, 12)) -> pd.DataFrame:
    """
    Build lag/rolling features per station:
      - rate of change (1h diff)
      - rolling std (volatility) over several windows
      - stuck-value run length (consecutive near-identical readings)
      - time since last valid reading
      - neighbor divergence (added separately, merged here)
    """
    df = df.sort_values(["station_id", "timestamp"]).copy()
    df = add_neighbor_divergence(df)

    feature_frames = []
    for stn, g in df.groupby("station_id", sort=False):
        g = g.sort_values("timestamp").copy()

        for var in ["pressure", "wind_speed"]:
            g[f"{var}_diff1h"] = g[var].diff()
            for w in windows:
                g[f"{var}_rollstd_{w}h"] = g[var].rolling(w, min_periods=2).std()
                g[f"{var}_rollmean_{w}h"] = g[var].rolling(w, min_periods=2).mean()

            # stuck-value detection: consecutive readings within a tiny tolerance
            tol = 0.02 if var == "pressure" else 0.1
            near_prev = (g[var].diff().abs() < tol).astype(int)
            # run-length of "stuck" behavior
            stuck_run = near_prev * (near_prev.groupby((near_prev != near_prev.shift()).cumsum()).cumcount() + 1)
            g[f"{var}_stuck_run"] = stuck_run

        # time since last valid (non-NaN) reading, in hours
        valid = g["pressure"].notna() & g["wind_speed"].notna()
        g["hours_since_valid"] = (
            g["timestamp"].where(valid).ffill()
            .rsub(g["timestamp"]).dt.total_seconds() / 3600
        )

        feature_frames.append(g)

    return pd.concat(feature_frames, ignore_index=True)


# ----------------------------------------------------------------------
# 3. LABEL CONSTRUCTION
# ----------------------------------------------------------------------

def build_labels(df: pd.DataFrame, lead_hours: int = 12, missing_col: str = "is_missing") -> pd.DataFrame:
    """
    For each station, find the first hour of sustained failure (missing_col
    becomes True and stays True), then label every row within `lead_hours`
    BEFORE that onset as a positive "about to fail" example (label=1).
    Rows already missing, or far from any failure, are label=0 (or dropped
    if you want a cleaner "normal vs about-to-fail" split -- see note below).
    """
    df = df.sort_values(["station_id", "timestamp"]).copy()
    df["label"] = 0
    df["hours_to_failure"] = np.nan

    for stn, g in df.groupby("station_id", sort=False):
        idx = g.index
        missing = g[missing_col].values
        if not missing.any():
            continue
        fail_idx_local = np.argmax(missing)  # first True
        onset_ts = g["timestamp"].iloc[fail_idx_local]

        window_start = onset_ts - pd.Timedelta(hours=lead_hours)
        pre_mask = (g["timestamp"] >= window_start) & (g["timestamp"] < onset_ts)
        df.loc[idx[pre_mask.values], "label"] = 1

        htf = (onset_ts - g["timestamp"]).dt.total_seconds() / 3600
        df.loc[idx, "hours_to_failure"] = htf.where(g["timestamp"] < onset_ts)

    return df


# ----------------------------------------------------------------------
# 4. MODELING
# ----------------------------------------------------------------------

FEATURE_COLS_TEMPLATE = [
    "pressure_diff1h", "wind_speed_diff1h",
    "pressure_rollstd_3h", "pressure_rollstd_6h", "pressure_rollstd_12h",
    "wind_speed_rollstd_3h", "wind_speed_rollstd_6h", "wind_speed_rollstd_12h",
    "pressure_stuck_run", "wind_speed_stuck_run",
    "pressure_neighbor_div", "wind_speed_neighbor_div",
    "hours_since_valid",
]


@dataclass
class TrainedModel:
    model: object
    feature_cols: list
    kind: str  # "isolation_forest" or "gradient_boosting"


def train_model(df: pd.DataFrame, kind: str = "gradient_boosting", feature_cols=None) -> TrainedModel:
    """
    kind = "isolation_forest": unsupervised anomaly detector, trained ONLY on
        rows that are NOT within the pre-failure window (i.e. "normal"
        behavior, including real storm conditions at non-failing stations).
        Good option if you don't have many labeled failure events yet.

    kind = "gradient_boosting": supervised classifier on label (0/1) built by
        build_labels(). Needs enough failure examples across your six
        hurricanes to be meaningful -- pool them together for training.
    """
    feature_cols = feature_cols or FEATURE_COLS_TEMPLATE
    data = df.dropna(subset=feature_cols).copy()

    if kind == "isolation_forest":
        normal = data[data["label"] == 0]
        model = IsolationForest(n_estimators=300, contamination="auto", random_state=0)
        model.fit(normal[feature_cols])
        return TrainedModel(model, feature_cols, kind)

    elif kind == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=0)
        model.fit(data[feature_cols], data["label"])
        return TrainedModel(model, feature_cols, kind)

    raise ValueError(f"Unknown kind: {kind}")


def score(df: pd.DataFrame, trained: TrainedModel) -> pd.DataFrame:
    """Attach a risk score (higher = more likely about to fail) to each row."""
    data = df.copy()
    valid = data.dropna(subset=trained.feature_cols)

    if trained.kind == "isolation_forest":
        raw = -trained.model.score_samples(valid[trained.feature_cols])  # higher = more anomalous
    else:
        raw = trained.model.predict_proba(valid[trained.feature_cols])[:, 1]

    data.loc[valid.index, "risk_score"] = raw
    return data


# ----------------------------------------------------------------------
# 5. EVALUATION
# ----------------------------------------------------------------------

def evaluate(df_scored: pd.DataFrame, risk_threshold: float = None, lead_hours: int = 12):
    """
    Reports:
      - lead time: for each station that fails, how many hours before onset
        did risk_score first cross the threshold?
      - false alarm rate: fraction of "normal" hours flagged as at-risk
      - average precision (area under PR curve) using `label` as ground truth
    """
    valid = df_scored.dropna(subset=["risk_score"])

    ap = average_precision_score(valid["label"], valid["risk_score"])
    print(f"Average precision (label vs risk_score): {ap:.3f}")

    if risk_threshold is None:
        # pick threshold at the 95th percentile of "normal" scores
        risk_threshold = valid.loc[valid.label == 0, "risk_score"].quantile(0.95)
    print(f"Using risk threshold: {risk_threshold:.4f}")

    false_alarm_rate = (
        valid.loc[valid.label == 0, "risk_score"] > risk_threshold
    ).mean()
    print(f"False alarm rate on normal hours: {false_alarm_rate:.3%}")

    lead_times = []
    for stn, g in valid.groupby("station_id"):
        g = g.sort_values("timestamp")
        pre_fail = g[g["hours_to_failure"].notna()]
        if pre_fail.empty:
            continue
        flagged = pre_fail[pre_fail["risk_score"] > risk_threshold]
        if not flagged.empty:
            first_flag_lead = flagged["hours_to_failure"].max()  # earliest = max hours-to-failure
            lead_times.append((stn, first_flag_lead))

    if lead_times:
        print("\nLead time before failure onset (first correct flag):")
        for stn, lt in lead_times:
            print(f"  {stn}: flagged {lt:.1f}h before failure")
        avg_lead = np.mean([lt for _, lt in lead_times])
        print(f"\nAverage lead time: {avg_lead:.1f}h (max possible = {lead_hours}h)")
    else:
        print("No failing stations were flagged before onset -- consider "
              "lowering the threshold or adding more training data.")

    precision, recall, _ = precision_recall_curve(valid["label"], valid["risk_score"])
    return {"average_precision": ap, "threshold": risk_threshold,
            "false_alarm_rate": false_alarm_rate, "lead_times": lead_times,
            "precision_curve": precision, "recall_curve": recall}


# ----------------------------------------------------------------------
# 6. LIVE / STREAMING SCORING
# ----------------------------------------------------------------------

class LiveStationMonitor:
    """
    Maintains a rolling buffer of recent readings for ONE live station and
    scores each new observation as it arrives -- e.g. call `ingest()` once
    per hour from a script that pulls the latest reading from a public feed.
    """

    def __init__(self, trained: TrainedModel, station_id: str, lat: float, lon: float,
                 buffer_hours: int = 24):
        self.trained = trained
        self.station_id = station_id
        self.lat = lat
        self.lon = lon
        self.buffer_hours = buffer_hours
        self.buffer = pd.DataFrame({
            "station_id": pd.Series(dtype="object"),
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "pressure": pd.Series(dtype="float64"),
            "wind_speed": pd.Series(dtype="float64"),
            "wind_dir": pd.Series(dtype="float64"),
            "lat": pd.Series(dtype="float64"),
            "lon": pd.Series(dtype="float64"),
        })

    def ingest(self, timestamp, pressure, wind_speed, wind_dir) -> Optional[float]:
        """Add a new reading, recompute features, and return the current risk score."""
        row = pd.DataFrame([{
            "station_id": self.station_id, "timestamp": pd.Timestamp(timestamp),
            "pressure": pressure, "wind_speed": wind_speed, "wind_dir": wind_dir,
            "lat": self.lat, "lon": self.lon,
        }])
        self.buffer = pd.concat([self.buffer, row], ignore_index=True)
        cutoff = self.buffer["timestamp"].max() - pd.Timedelta(hours=self.buffer_hours)
        self.buffer = self.buffer[self.buffer["timestamp"] >= cutoff].reset_index(drop=True)

        if len(self.buffer) < 4:
            return None  # not enough history yet for rolling features

        feats = engineer_features(self.buffer)
        latest = feats.iloc[[-1]]
        if latest[self.trained.feature_cols].isna().any(axis=1).iloc[0]:
            return None

        if self.trained.kind == "isolation_forest":
            return float(-self.trained.model.score_samples(latest[self.trained.feature_cols])[0])
        else:
            return float(self.trained.model.predict_proba(latest[self.trained.feature_cols])[0, 1])


# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 1. Loading data (synthetic demo) ===")
    raw = load_station_data()
    print(raw.groupby("station_id")["is_missing"].any())

    print("\n=== 2. Engineering features ===")
    feats = engineer_features(raw)

    print("\n=== 3. Building labels (12h pre-failure window) ===")
    labeled = build_labels(feats, lead_hours=12)
    print("Positive (about-to-fail) rows:", (labeled.label == 1).sum())

    print("\n=== 4. Training model (gradient boosting) ===")
    trained = train_model(labeled, kind="gradient_boosting")

    print("\n=== 5. Scoring + evaluating ===")
    scored = score(labeled, trained)
    results = evaluate(scored, lead_hours=12)

    print("\n=== 6. Live monitor demo (feeding one station hour by hour) ===")
    monitor = LiveStationMonitor(trained, station_id="STN_A_coastal_fails", lat=40.6, lon=-73.9)
    live_station_data = raw[raw.station_id == "STN_A_coastal_fails"].sort_values("timestamp")
    for _, r in live_station_data.iloc[150:175].iterrows():
        risk = monitor.ingest(r.timestamp, r.pressure, r.wind_speed, r.wind_dir)
        flag = "  <-- AT RISK" if (risk is not None and risk > results["threshold"]) else ""
        print(f"{r.timestamp}: risk={risk}{flag}")
