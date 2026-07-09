"""
isd_loader.py

Downloads and parses real NOAA ISD-Lite hourly station data for a chosen
region and date range, and reshapes it into the schema expected by
station_failure_prediction.py:

    station_id | timestamp | pressure | wind_speed | wind_dir | lat | lon | is_missing

Data source: NOAA Integrated Surface Database, Lite format
  - Station list:  https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv
  - Hourly files:  https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{USAF}-{WBAN}-{year}.gz

ISD-Lite is a fixed-width format with 8 quality-controlled fields per hour
(temperature, dew point, sea level pressure, wind direction, wind speed,
sky cover, and two precip windows). We only need pressure and wind here.
Format spec: https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.txt

Usage
-----
    from isd_loader import get_stations_in_box, load_isd_lite_region

    stations = get_stations_in_box(lat_min=38.8, lat_max=41.5,
                                    lon_min=-75.5, lon_max=-71.5)

    df = load_isd_lite_region(stations, years=[2012],
                               start="2012-10-20", end="2012-11-05")

    # df is now ready for:
    #   from station_failure_prediction import engineer_features, build_labels, ...
    #   feats = engineer_features(df)
"""

import gzip
import io
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_LITE_URL_TMPL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz"

# ISD-Lite fixed-width column spec (0-indexed, end-exclusive), per NOAA's
# isd-lite-format.txt. Values are scaled integers; divide by SCALE to get
# real units. -9999 means missing.
_COLSPECS = [
    (0, 4),    # year
    (5, 7),    # month
    (8, 10),   # day
    (11, 13),  # hour
    (13, 19),  # air temp (C * 10)
    (19, 25),  # dew point (C * 10)
    (25, 31),  # sea level pressure (hPa * 10)
    (31, 37),  # wind direction (degrees)
    (37, 43),  # wind speed (m/s * 10)
    (43, 49),  # sky cover code
    (49, 55),  # precip 1h (mm * 10)
    (55, 61),  # precip 6h (mm * 10)
]
_COLNAMES = ["year", "month", "day", "hour", "air_temp", "dew_point",
             "slp", "wind_dir", "wind_speed", "sky_cover", "precip_1h", "precip_6h"]


# ----------------------------------------------------------------------
# STATION METADATA
# ----------------------------------------------------------------------

def get_stations_in_box(lat_min: float, lat_max: float,
                         lon_min: float, lon_max: float,
                         min_years_active: int = 1) -> pd.DataFrame:
    """
    Download NOAA's global station list and return stations whose lat/lon
    fall in the given bounding box. Use the same box you already use for
    filtering GHCNd stations for a given hurricane.

    Returns columns: usaf, wban, station_name, lat, lon, elev, begin, end
    """
    resp = requests.get(ISD_HISTORY_URL, timeout=60)
    resp.raise_for_status()
    hist = pd.read_csv(io.StringIO(resp.text), dtype={"USAF": str, "WBAN": str})

    hist = hist.rename(columns={
        "USAF": "usaf", "WBAN": "wban", "STATION NAME": "station_name",
        "LAT": "lat", "LON": "lon", "ELEV(M)": "elev",
        "BEGIN": "begin", "END": "end",
    })

    box = hist[
        hist["lat"].between(lat_min, lat_max) &
        hist["lon"].between(lon_min, lon_max)
    ].copy()

    box = box.dropna(subset=["lat", "lon"])
    box["begin"] = pd.to_numeric(box["begin"], errors="coerce")
    box["end"] = pd.to_numeric(box["end"], errors="coerce")
    box = box.dropna(subset=["begin", "end"])

    return box.reset_index(drop=True)


# ----------------------------------------------------------------------
# DOWNLOAD + PARSE ONE STATION-YEAR
# ----------------------------------------------------------------------

def _download_station_year(usaf: str, wban: str, year: int) -> Optional[pd.DataFrame]:
    """Download and parse one ISD-Lite station-year file. Returns None if unavailable."""
    url = ISD_LITE_URL_TMPL.format(year=year, usaf=usaf, wban=wban)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None
        raw = gzip.decompress(resp.content).decode("utf-8", errors="ignore")
    except Exception:
        return None

    lines = [ln for ln in raw.splitlines() if len(ln) >= 61]
    if not lines:
        return None

    records = []
    for ln in lines:
        vals = [ln[start:end].strip() for start, end in _COLSPECS]
        records.append(vals)

    df = pd.DataFrame(records, columns=_COLNAMES)
    df = df.apply(pd.to_numeric, errors="coerce")

    df["timestamp"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day, hour=df.hour),
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])

    # apply scaling factors and missing-value codes (-9999 -> NaN)
    for col, scale in [("air_temp", 10), ("dew_point", 10), ("slp", 10), ("wind_speed", 10)]:
        df[col] = df[col].replace(-9999, np.nan) / scale
    df["wind_dir"] = df["wind_dir"].replace(-9999, np.nan)

    df["station_id"] = f"{usaf}-{wban}"
    return df[["station_id", "timestamp", "slp", "wind_speed", "wind_dir"]].rename(
        columns={"slp": "pressure"}
    )


# ----------------------------------------------------------------------
# REGION LOADER
# ----------------------------------------------------------------------

def load_isd_lite_region(stations: pd.DataFrame, years: Iterable[int],
                          start: str, end: str,
                          max_stations: Optional[int] = None,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Download ISD-Lite data for every station in `stations` (as returned by
    get_stations_in_box) across the given years, clip to [start, end], and
    reshape into the pipeline's expected schema -- including reindexing
    each station to a complete hourly timeline so that true reporting gaps
    show up as NaN rows (this is what becomes `is_missing`).

    max_stations caps how many stations to pull, useful for a quick test
    run before downloading a whole region.
    """
    station_list = stations if max_stations is None else stations.head(max_stations)

    all_frames = []
    full_index = pd.date_range(start, end, freq="h")

    for _, row in station_list.iterrows():
        usaf, wban = str(row.usaf).zfill(6), str(row.wban).zfill(5)
        frames = []
        for year in years:
            df_year = _download_station_year(usaf, wban, year)
            if df_year is not None:
                frames.append(df_year)
        if not frames:
            if verbose:
                print(f"  [skip] {usaf}-{wban}: no data available")
            continue

        df_stn = pd.concat(frames, ignore_index=True)
        df_stn = df_stn[(df_stn.timestamp >= start) & (df_stn.timestamp <= end)]
        if df_stn.empty:
            continue

        # Reindex to a complete hourly grid so real gaps become visible NaNs
        # rather than silently-absent rows.
        df_stn = df_stn.set_index("timestamp").reindex(full_index)
        df_stn.index.name = "timestamp"
        df_stn = df_stn.reset_index()
        df_stn["station_id"] = f"{usaf}-{wban}"
        df_stn["lat"] = row.lat
        df_stn["lon"] = row.lon

        # is_missing: no pressure AND no wind for that hour
        df_stn["is_missing"] = df_stn["pressure"].isna() & df_stn["wind_speed"].isna()

        all_frames.append(df_stn)
        if verbose:
            pct_missing = df_stn["is_missing"].mean() * 100
            print(f"  [ok]   {usaf}-{wban} ({row.station_name}): "
                  f"{len(df_stn)} hours, {pct_missing:.1f}% missing")

    if not all_frames:
        raise RuntimeError("No station data could be downloaded for this region/period. "
                            "Check the bounding box and year range.")

    result = pd.concat(all_frames, ignore_index=True)
    return result[["station_id", "timestamp", "pressure", "wind_speed",
                    "wind_dir", "lat", "lon", "is_missing"]]


# ----------------------------------------------------------------------
# DEMO: Hurricane Sandy region, small subset
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Fetching station list for the NYC / Sandy region ===")
    stations = get_stations_in_box(lat_min=39.5, lat_max=41.5, lon_min=-75.0, lon_max=-72.0)
    print(f"Found {len(stations)} candidate stations in the box.")

    print("\n=== Downloading ISD-Lite data (first 5 stations, Oct 2012) ===")
    df = load_isd_lite_region(
        stations,
        years=[2012],
        start="2012-10-20",
        end="2012-11-05",
        max_stations=5,
    )

    print("\nSample of loaded data:")
    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")
    print(df.groupby("station_id")["is_missing"].mean())

    df.to_csv("sandy_isd_sample.csv", index=False)
    print("\nSaved to sandy_isd_sample.csv -- feed this into "
          "station_failure_prediction.load_station_data(path=...)")
