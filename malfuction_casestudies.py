import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from math import radians, sin, cos, sqrt, atan2
import glob


os.chdir(os.path.dirname(os.path.abspath(__file__)))


USE_GEOPANDAS = True
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    USE_GEOPANDAS = False
    print("GeoPandas not available")

USE_SEABORN = True
try:
    import seaborn as sns
except Exception:
    USE_SEABORN = False
    print("Seaborn not available")

USE_SCIPY = True
try:
    from scipy.stats import (f_oneway, kruskal, ttest_ind, mannwhitneyu,
                              chi2_contingency, spearmanr, kendalltau, ttest_rel, levene)
    from scipy.interpolate import griddata
except Exception:
    USE_SCIPY = False
    print("SciPy not available")


# CASE STUDIES

case_studies = [
    {"name": "Sandy",
     "lat_min": 39.5, "lat_max": 42.5, "lon_min": -75.5, "lon_max": -72.0,
     "start": "2012-10-25", "end": "2012-11-05",
     "prev_start": "2011-10-25", "prev_end": "2011-11-05"},
    {"name": "Ian",
     "lat_min": 26.0, "lat_max": 34.0, "lon_min": -84.0, "lon_max": -78.5,
     "start": "2022-09-25", "end": "2022-10-05",
     "prev_start": "2021-09-25", "prev_end": "2021-10-05"},
    {"name": "Harvey",
     "lat_min": 27.0, "lat_max": 31.5, "lon_min": -96.5, "lon_max": -91.0,
     "start": "2017-08-23", "end": "2017-09-05",
     "prev_start": "2016-08-23", "prev_end": "2016-09-05"},
    {"name": "Rita",
     "lat_min": 26.0, "lat_max": 33.5, "lon_min": -98.5, "lon_max": -87.0,
     "start": "2005-09-18", "end": "2005-09-26",
     "prev_start": "2004-09-18", "prev_end": "2004-09-26"},
    {"name": "Michael",
     "lat_min": 29.0, "lat_max": 33.0, "lon_min": -86.5, "lon_max": -83.0,
     "start": "2018-10-07", "end": "2018-10-12",
     "prev_start": "2017-10-07", "prev_end": "2017-10-12"},
]

SPATIAL_RADIUS_KM        = 100    # neighbours must be within this distance to be used
MIN_NEIGHBORS            = 3      # need at least this many neighbours reporting that day
MAD_Z_THRESHOLD          = 3.5    # robust z-score (MAD-based) beyond which a reading is "Suspect"
MALFUNCTION_SUSPECT_PCT  = 20.0   # suspect% at/above this flags a station "Likely_Malfunctioning"


# USER VARIABLE
variable = input("Enter variable (TMAX, TMIN, PRCP, TAVG): ").strip().upper()

# FILES
station_file   = 'ghcnd-stations.txt'
inventory_file = 'ghcnd-inventory.txt'
dly_folder     = 'dly_files'
out_root       = 'malfunction'
os.makedirs(dly_folder, exist_ok=True)
os.makedirs(out_root,   exist_ok=True)

for req in [station_file, inventory_file]:
    if not os.path.exists(req):
        print(f"Missing required file: {req}\n"
              f"Download from https://www.ncei.noaa.gov/pub/data/ghcn/daily/")
        raise SystemExit

# Load inventory + stations
print("Loading station inventory...")
inventory = pd.read_csv(
    inventory_file, sep=r'\s+', header=None,
    names=["ID", "LAT", "LON", "ELEMENT", "FIRSTYEAR", "LASTYEAR"],
    engine='python'
)
inventory['ID'] = inventory['ID'].astype(str).str.strip()

station_data = []
with open(station_file, 'r') as fh:
    for line in fh:
        sid  = line[0:11].strip()
        lat  = float(line[12:20].strip())
        lon  = float(line[21:30].strip())
        name = line[41:71].strip()
        station_data.append((sid, lat, lon, name))
station_df = pd.DataFrame(station_data, columns=['ID', 'LAT', 'LON', 'NAME'])
station_df['ID'] = station_df['ID'].astype(str).str.strip()
print(f"Loaded {len(station_df)} stations.")

# parse dly
def parse_dly(filepath, variable, start_date, end_date):
    days_dict = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line[17:21] == variable:
                    year  = int(line[11:15])
                    month = int(line[15:17])
                    for i in range(31):
                        try:
                            val_str = line[21 + i * 8: 26 + i * 8]
                            val = int(val_str[:5])
                            d = date(year, month, i + 1)
                            if start_date <= d <= end_date and val != -9999:
                                days_dict[d] = val
                        except Exception:
                            continue
    except Exception as e:
        print(f"  parse error ({filepath}): {e}")
    return days_dict

def download_dly(sid, base_url, dest, retries=3, timeout=45):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{base_url}/{sid}.dly", timeout=timeout)
            if r.status_code == 200:
                with open(dest, 'wb') as fh:
                    fh.write(r.content)
                return True
            else:
                print(f"  [{sid}] HTTP {r.status_code} on attempt {attempt}")
        except requests.exceptions.ConnectionError as e:
            print(f"  [{sid}] Connection error attempt {attempt}: {e}")
        except requests.exceptions.Timeout:
            print(f"  [{sid}] Timeout on attempt {attempt}")
        except Exception as e:
            print(f"  [{sid}] Unexpected error attempt {attempt}: {e}")
    print(f"  [{sid}] Giving up after {retries} attempts — skipping.")
    return False

#GIS
def haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a    = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c    = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

COAST_POINTS = [
    (47.6, -122.3), (37.8, -122.4), (34.0, -118.2), (32.7, -117.1),
    (29.76, -95.37), (30.3, -81.4), (25.8, -80.2), (27.8, -82.6),
    (30.2, -89.2), (40.7, -74.0), (39.3, -76.6), (36.85, -75.98), (43.65, -70.25),
]

CITY_CENTROIDS = [
    ("New York",      40.7128, -74.0060),
    ("Philadelphia",  39.9526, -75.1652),
    ("Baltimore",     39.2904, -76.6122),
    ("Boston",        42.3601, -71.0589),
    ("Miami",         25.7617, -80.1918),
    ("Tampa",         27.9506, -82.4572),
    ("Jacksonville",  30.3322, -81.6557),
    ("Houston",       29.7604, -95.3698),
    ("New Orleans",   29.9511, -90.0715),
    ("Tallahassee",   30.4383, -84.2807),
    ("Atlanta",       33.7490, -84.3880),
]

def download_and_load_naturalearth():
    try:
        coast_url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
        urban_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_urban_areas.zip"
        tmpdir = "temp_naturalearth"
        os.makedirs(tmpdir, exist_ok=True)

        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36")
        }

        print("  Downloading Natural Earth coastline...")
        r = requests.get(coast_url, headers=headers, timeout=60)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(tmpdir)
        coast_shps = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.shp')]
        gdf_coast  = gpd.read_file(coast_shps[0])

        print("  Downloading Natural Earth urban areas...")
        r2 = requests.get(urban_url, headers=headers, timeout=60)
        r2.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r2.content)).extractall(tmpdir)
        urban_shps = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
                      if f.endswith('.shp') and 'urban' in f.lower()]
        if not urban_shps:
            urban_shps = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.shp')]
        gdf_urban = gpd.read_file(urban_shps[0])
        return gdf_coast, gdf_urban
    except Exception as e:
        print(f"  Natural Earth download/load failed: {e}")
        print("  Using haversine fallback for region assignment.")
        return None, None

def compute_distances_and_regions(stations_df, output_folder, coastal_km_thresh=50):
    st = stations_df.copy().reset_index(drop=True)
    if USE_GEOPANDAS:
        try:
            gdf_coast, gdf_urban = download_and_load_naturalearth()
            if gdf_coast is not None:
                gdf_coast_proj = gdf_coast.to_crs(epsg=3857)
                try:
                    coast_union = gdf_coast_proj.union_all()   # newer geopandas/shapely
                except AttributeError:
                    coast_union = gdf_coast_proj.unary_union   # older geopandas/shapely
                pts      = gpd.GeoDataFrame(
                    st,
                    geometry=[Point(xy) for xy in zip(st['LON'], st['LAT'])],
                    crs="EPSG:4326"
                )
                pts_proj = pts.to_crs(epsg=3857)
                dists_m  = pts_proj.geometry.apply(lambda p: p.distance(coast_union))
                distances = dists_m.values / 1000.0

                if gdf_urban is not None:
                    gdf_urban_proj = gdf_urban.to_crs(epsg=3857)
                    try:
                        urban_union = gdf_urban_proj.union_all()
                    except AttributeError:
                        urban_union = gdf_urban_proj.unary_union
                    pts_proj['in_urban'] = pts_proj.within(urban_union)
                    in_urban = pts_proj['in_urban'].values
                else:
                    in_urban = np.array([False] * len(st))

                regions = []
                for i, d in enumerate(distances):
                    if d <= coastal_km_thresh:
                        regions.append('Coastal')
                    elif in_urban[i]:
                        regions.append('Inland')
                    else:
                        regions.append('Rural')

                st['Distance_to_Coast_km'] = distances
                st['Region'] = regions
                return st
        except Exception as e:
            print(f"  GeoPandas method failed, falling back: {e}")

    # Fallback: haversine
    dist_to_coast = []
    is_urban      = []
    for _, row in st.iterrows():
        lat, lon = row['LAT'], row['LON']
        min_d = min(haversine_km(lat, lon, cp[0], cp[1]) for cp in COAST_POINTS)
        dist_to_coast.append(min_d)
        urb = any(haversine_km(lat, lon, c[1], c[2]) <= 30 for c in CITY_CENTROIDS)
        is_urban.append(urb)

    st['Distance_to_Coast_km'] = np.array(dist_to_coast)
    regions = []
    for i, d in enumerate(dist_to_coast):
        if d <= coastal_km_thresh:
            regions.append('Coastal')
        elif is_urban[i]:
            regions.append('Inland')
        else:
            regions.append('Rural')
    st['Region'] = regions

    pd.DataFrame({
        'ID': st['ID'],
        'Distance_to_Coast_km': st['Distance_to_Coast_km'],
        'Region': st['Region'],
    }).to_csv(os.path.join(output_folder, 'region_assignment_fallback.csv'), index=False)
    return st


# SPATIAL MALFUNCTION / SUSPECT-DATA DETECTION
#
# A station can be "malfunctioning" while it keeps reporting numbers — e.g.
# a stuck sensor, a decimal/unit error, a miscalibrated gauge. Those look
# fine at a glance (a value exists) but disagree badly with everything
# nearby was reporting that same day. compute_spatial_anomalies() estimates
# what each station "should" have read from its neighbours and flags days
# where the actual reading is a statistical outlier against that estimate.

def compute_spatial_anomalies(station_vals, stations_df, date_list,
                               radius_km=SPATIAL_RADIUS_KM,
                               min_neighbors=MIN_NEIGHBORS,
                               mad_z_thresh=MAD_Z_THRESHOLD):
    """
    station_vals: dict {ID: {date: value}} as returned by parse_dly per station.
    stations_df:  DataFrame with ID, LAT, LON for the same stations.
    date_list:    list of date objects covering the period of interest.

    Returns a long-format DataFrame with one row per station-day that had
    enough nearby data to be checked: Date, ID, Value, Neighbor_Estimate,
    Residual, Robust_Z, Suspect.
    """
    coords = stations_df.drop_duplicates('ID').set_index('ID')[['LAT', 'LON']]
    sids   = [sid for sid in station_vals.keys() if sid in coords.index]

    dist_cache = {}
    def get_dist(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in dist_cache:
            la, lo  = coords.loc[a, 'LAT'], coords.loc[a, 'LON']
            lb, lob = coords.loc[b, 'LAT'], coords.loc[b, 'LON']
            dist_cache[key] = haversine_km(la, lo, lb, lob)
        return dist_cache[key]

    rows = []
    for d in date_list:
        day_vals = {sid: station_vals[sid][d] for sid in sids
                    if d in station_vals.get(sid, {})}
        if len(day_vals) < min_neighbors + 1:
            continue

        residuals  = {}
        estimates  = {}
        for sid, val in day_vals.items():
            neighbors = []
            for other_sid, other_val in day_vals.items():
                if other_sid == sid:
                    continue
                dist = get_dist(sid, other_sid)
                if dist <= radius_km:
                    neighbors.append((dist, other_val))
            if len(neighbors) < min_neighbors:
                continue
            weights = [1.0 / max(dist, 0.5) for dist, _ in neighbors]
            vals_n  = [v for _, v in neighbors]
            est = sum(w * v for w, v in zip(weights, vals_n)) / sum(weights)
            estimates[sid] = est
            residuals[sid] = val - est

        if len(residuals) < 3:
            continue

        res_arr    = np.array(list(residuals.values()), dtype=float)
        median_res = np.median(res_arr)
        mad        = np.median(np.abs(res_arr - median_res))
        mad        = mad if mad > 1e-6 else (np.std(res_arr) + 1e-6)

        for sid, resid in residuals.items():
            robust_z = 0.6745 * (resid - median_res) / mad
            rows.append({
                'Date':               d.strftime("%Y-%m-%d"),
                'ID':                 sid,
                'Value':              day_vals[sid],
                'Neighbor_Estimate':  round(estimates[sid], 2),
                'Residual':           round(resid, 2),
                'Robust_Z':           round(robust_z, 2),
                'Suspect':            bool(abs(robust_z) > mad_z_thresh),
            })

    return pd.DataFrame(rows)


def summarize_malfunction(stations_base_df, anomaly_df,
                           suspect_thresh=MALFUNCTION_SUSPECT_PCT):
    """
    Builds the per-station malfunction picture purely from spatial-outlier
    ("Suspect") behaviour: how many days a station could be checked against
    its neighbours, and how many of those days its reading disagreed badly
    with the neighbourhood estimate.
    """
    if anomaly_df.empty:
        suspect_summary = pd.DataFrame(columns=['ID', 'Days_Checked', 'Suspect_Days'])
    else:
        suspect_summary = (anomaly_df.groupby('ID')
                            .agg(Days_Checked=('Suspect', 'size'),
                                 Suspect_Days=('Suspect', 'sum'))
                            .reset_index())

    merged = stations_base_df.merge(suspect_summary, on='ID', how='left')
    merged['Days_Checked'] = merged['Days_Checked'].fillna(0).astype(int)
    merged['Suspect_Days'] = merged['Suspect_Days'].fillna(0).astype(int)
    merged['Suspect_%']    = np.where(
        merged['Days_Checked'] > 0,
        np.round(100 * merged['Suspect_Days'] / merged['Days_Checked'].replace(0, np.nan), 2),
        0.0
    )
    merged['Suspect_%'] = merged['Suspect_%'].fillna(0.0)

    merged['Likely_Malfunctioning'] = merged['Suspect_%'] >= suspect_thresh
    # With only one symptom being tracked (spatial disagreement), the score
    # is just the Suspect_% itself.
    merged['Malfunction_Score'] = merged['Suspect_%']

    return merged


# run_event_period
def run_event_period(event_name, variable, lat_min, lat_max, lon_min, lon_max, start_str, end_str):
    print(f"\n--- Running: {event_name} | {start_str} to {end_str} ---")

    output_folder = os.path.join(out_root, event_name, f"{start_str}_to_{end_str}")
    os.makedirs(output_folder, exist_ok=True)
    plots_folder  = os.path.join(output_folder, "plots")
    tables_folder = os.path.join(output_folder, "tables")
    os.makedirs(plots_folder,  exist_ok=True)
    os.makedirs(tables_folder, exist_ok=True)

    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date   = datetime.strptime(end_str,   "%Y-%m-%d").date()

    inv_filtered = inventory[
        (inventory['ELEMENT']   == variable) &
        (inventory['FIRSTYEAR'] <= start_date.year) &
        (inventory['LASTYEAR']  >= end_date.year)
    ]

    stations = station_df[
        (station_df['LAT'] >= lat_min) & (station_df['LAT'] <= lat_max) &
        (station_df['LON'] >= lon_min) & (station_df['LON'] <= lon_max)
    ]
    stations = pd.merge(stations, inv_filtered[['ID']], on='ID', how='inner')
    print(f"  Stations in region matching {variable}: {len(stations)}")

    base_url    = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all"
    unique_sids = stations['ID'].unique()
    need_download = [sid for sid in unique_sids
                     if not os.path.exists(os.path.join(dly_folder, f"{sid}.dly"))]
    print(f"  Downloading {len(need_download)} new .dly files "
          f"({len(unique_sids) - len(need_download)} already cached)...")

    for i, sid in enumerate(need_download, 1):
        dest = os.path.join(dly_folder, f"{sid}.dly")
        if (i % 20 == 0) or (i == len(need_download)):
            print(f"    {i}/{len(need_download)} downloaded...")
        download_dly(sid, base_url, dest)

    date_list = [start_date + timedelta(days=i)
                           for i in range((end_date - start_date).days + 1)]

    print("  Parsing station files...")
    station_vals = {}
    for sid in unique_sids:
        path = os.path.join(dly_folder, f"{sid}.dly")
        if os.path.exists(path):
            station_vals[sid] = parse_dly(path, variable, start_date, end_date)

    print("  Computing distances and regions...")
    region_df = compute_distances_and_regions(
        stations[['ID', 'LAT', 'LON', 'NAME']], output_folder, coastal_km_thresh=50)

    print("  Checking readings against neighbouring stations...")
    anomaly_df = compute_spatial_anomalies(
        station_vals, stations[['ID', 'LAT', 'LON']], date_list)
    anomaly_df.to_csv(os.path.join(output_folder, 'spatial_anomalies_detail.csv'), index=False)

    malfunction_df = summarize_malfunction(
        region_df[['ID', 'LAT', 'LON', 'NAME', 'Distance_to_Coast_km', 'Region']], anomaly_df)
    malfunction_df.to_csv(
        os.path.join(output_folder, 'stations_malfunction_summary.csv'), index=False)
    n_flagged = int(malfunction_df['Likely_Malfunctioning'].sum())
    print(f"  Flagged {n_flagged}/{len(malfunction_df)} stations as Likely_Malfunctioning "
          f"(suspect >= {MALFUNCTION_SUSPECT_PCT}% of checked days).")

    # stats
    out_lines = [
        f"Event: {event_name} {start_str} to {end_str}",
        f"Stations considered: {len(malfunction_df)}",
        f"Stations flagged Likely_Malfunctioning (spatially suspect readings): {n_flagged}",
        "",
    ]

    if len(malfunction_df) >= 3:
        corr_lat = malfunction_df['LAT'].corr(malfunction_df['Suspect_%'])
        corr_lon = malfunction_df['LON'].corr(malfunction_df['Suspect_%'])
        out_lines.append(
            f"Pearson correlation: LAT vs Suspect% = {corr_lat:.3f}; "
            f"LON vs Suspect% = {corr_lon:.3f}"
        )
        if USE_SCIPY:
            rho, p_rho = spearmanr(
                malfunction_df['Distance_to_Coast_km'], malfunction_df['Suspect_%'])
            tau, p_tau = kendalltau(
                malfunction_df['Distance_to_Coast_km'], malfunction_df['Suspect_%'])
            out_lines.append(f"Spearman (distance->suspect) rho={rho:.3f}, p={p_rho:.3f}")
            out_lines.append(f"Kendall tau={tau:.3f}, p={p_tau:.3f}")
    else:
        out_lines.append("Not enough stations for correlations.")

    region_stats = (malfunction_df
                    .groupby('Region')['Suspect_%']
                    .agg(['mean', 'std', 'count'])
                    .reset_index())
    region_stats.to_csv(os.path.join(tables_folder, 'region_suspect_stats.csv'), index=False)

    test_results = []
    groups = {
        r: malfunction_df[malfunction_df['Region'] == r]['Suspect_%'].dropna().values
        for r in malfunction_df['Region'].unique()
    }

    if USE_SCIPY and len(groups) >= 2 and all(len(v) >= 2 for v in groups.values()):
        try:
            anova_stat, anova_p = f_oneway(*groups.values())
            test_results.append({'test': 'ANOVA', 'stat': anova_stat, 'p': anova_p})
        except Exception:
            pass
        try:
            kw_stat, kw_p = kruskal(*groups.values())
            test_results.append({'test': 'Kruskal', 'stat': kw_stat, 'p': kw_p})
        except Exception:
            pass

    if 'Coastal' in groups and 'Rural' in groups:
        g_c, g_r = groups['Coastal'], groups['Rural']
        if USE_SCIPY and len(g_c) >= 2 and len(g_r) >= 2:
            t_s, t_p = ttest_ind(g_c, g_r, equal_var=False)
            test_results.append({'test': 'Ttest_Coastal_vs_Rural',      'stat': t_s, 'p': t_p})
            m_s, m_p = mannwhitneyu(g_c, g_r, alternative='two-sided')
            test_results.append({'test': 'MannWhitney_Coastal_vs_Rural', 'stat': m_s, 'p': m_p})

    if 'Coastal' in groups and 'Inland' in groups:
        g_c, g_i = groups['Coastal'], groups['Inland']
        if USE_SCIPY and len(g_c) >= 2 and len(g_i) >= 2:
            t_s2, t_p2 = ttest_ind(g_c, g_i, equal_var=False)
            test_results.append({'test': 'Ttest_Coastal_vs_Inland',      'stat': t_s2, 'p': t_p2})
            m_s2, m_p2 = mannwhitneyu(g_c, g_i, alternative='two-sided')
            test_results.append({'test': 'MannWhitney_Coastal_vs_Inland', 'stat': m_s2, 'p': m_p2})

    try:
        median_suspect = malfunction_df['Suspect_%'].median()
        malfunction_df['SuspectCat'] = np.where(
            malfunction_df['Suspect_%'] > median_suspect, 'High', 'Low')
        contingency = pd.crosstab(malfunction_df['Region'], malfunction_df['SuspectCat'])
        if USE_SCIPY and contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_chi, dof, exp = chi2_contingency(contingency)
            test_results.append(
                {'test': 'ChiSquare_region_vs_suspectcat', 'stat': chi2, 'p': p_chi})
    except Exception:
        pass

    tests_df = pd.DataFrame(test_results)
    tests_df.to_csv(os.path.join(tables_folder, 'statistical_tests_results.csv'), index=False)

    print("  Generating plots...")
    if not malfunction_df.empty:
        if USE_SEABORN:
            sns.set(style="whitegrid")
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=malfunction_df, x="Region", y="Suspect_%")
            plt.title(f"{event_name}: Suspect % by Region")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_folder, 'boxplot_suspect_by_region.png'))
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.violinplot(data=malfunction_df, x="Region", y="Suspect_%")
            plt.title(f"{event_name}: Suspect % distribution by Region")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_folder, 'violin_suspect_by_region.png'))
            plt.close()
        else:
            plt.figure(figsize=(8, 6))
            malfunction_df.boxplot(column='Suspect_%', by='Region')
            plt.title(f"{event_name}: Suspect % by Region")
            plt.suptitle("")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_folder, 'boxplot_suspect_by_region.png'))
            plt.close()

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            malfunction_df['LON'], malfunction_df['LAT'],
            c=malfunction_df['Suspect_%'], s=40, cmap='Purples',
            edgecolor='k', linewidth=0.2
        )
        plt.colorbar(sc, label='% Days Flagged Suspect (vs neighbours)')
        plt.title(f"{event_name}: Spatial Outlier / Suspect Reading %")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, 'spatial_scatter_suspect.png'))
        plt.close()

        try:
            if USE_SCIPY and len(malfunction_df) >= 4:
                xi = np.linspace(malfunction_df['LON'].min(),
                                 malfunction_df['LON'].max(), 200)
                yi = np.linspace(malfunction_df['LAT'].min(),
                                 malfunction_df['LAT'].max(), 200)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = griddata(
                    (malfunction_df['LON'], malfunction_df['LAT']),
                    malfunction_df['Suspect_%'],
                    (Xi, Yi), method='linear'
                )
                plt.figure(figsize=(8, 6))
                plt.contourf(Xi, Yi, Zi, levels=14, cmap='Purples', alpha=0.8)
                plt.scatter(
                    malfunction_df['LON'], malfunction_df['LAT'],
                    c=malfunction_df['Suspect_%'], s=20, cmap='Purples',
                    edgecolor='k', linewidth=0.2
                )
                plt.colorbar(label='% Days Flagged Suspect')
                plt.title(f"{event_name}: Interpolated Heatmap of Suspect %")
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_folder, 'heatmap_interpolated_suspect.png'))
                plt.close()
        except Exception:
            pass

    if not anomaly_df.empty:
        plt.figure(figsize=(8, 6))
        plt.hist(anomaly_df['Robust_Z'], bins=40, color='slateblue', edgecolor='k', alpha=0.8)
        plt.axvline(MAD_Z_THRESHOLD,  color='red', linestyle='--', linewidth=1)
        plt.axvline(-MAD_Z_THRESHOLD, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Robust Z-score (reading vs. neighbourhood estimate)')
        plt.ylabel('Station-days')
        plt.title(f"{event_name}: Distribution of Spatial Residual Z-scores")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, 'residual_zscore_histogram.png'))
        plt.close()

    with open(os.path.join(output_folder, 'interpretation_summary.txt'), 'w') as fh:
        fh.write("\n".join(out_lines))

    print(f"  Done: {event_name} {start_str} to {end_str}")
    return malfunction_df, anomaly_df, region_stats, tests_df, plots_folder, tables_folder

# network resilience metrics
def compute_network_metrics(before_df, after_df):
    def failure_mask(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=['ID', 'LAT', 'LON', 'Failed'])
        m = df.copy()
        m['Failed'] = m['Likely_Malfunctioning']
        return m[['ID', 'LAT', 'LON', 'Failed']]

    b = failure_mask(before_df)
    a = failure_mask(after_df)

    def coverage(df):
        if df is None or df.empty:
            return np.nan
        return df['Failed'].sum() / len(df)

    def redundancy(df):
        if df is None or df.empty:
            return np.nan
        total  = len(df)
        failed = df['Failed'].sum()
        if failed == 0:
            return float('inf')
        return total / failed

    def mean_nearest_distance(df):
        if df is None or df.empty:
            return np.nan
        failed  = df[df['Failed']]
        working = df[~df['Failed']]
        if failed.empty or working.empty:
            return np.nan
        dlist = []
        for _, f in failed.iterrows():
            dmin = working.apply(
                lambda r: haversine_km(f['LAT'], f['LON'], r['LAT'], r['LON']), axis=1
            ).min()
            dlist.append(dmin)
        return float(np.mean(dlist)) if dlist else np.nan

    cov_b = coverage(b);               cov_a = coverage(a)
    red_b = redundancy(b);             red_a = redundancy(a)
    mnd_b = mean_nearest_distance(b);  mnd_a = mean_nearest_distance(a)

    def safe_delta(x, y):
        if (x is None or y is None or
                (isinstance(x, float) and np.isnan(x)) or
                (isinstance(y, float) and np.isnan(y)) or
                x == float('inf') or y == float('inf')):
            return np.nan
        return y - x

    return {
        'coverage_before':                  cov_b,
        'coverage_after':                   cov_a,
        'redundancy_before':                red_b,
        'redundancy_after':                 red_a,
        'mean_nearest_distance_before_km':  mnd_b,
        'mean_nearest_distance_after_km':   mnd_a,
        'coverage_delta':                   safe_delta(cov_b, cov_a),
        'redundancy_delta':                 safe_delta(red_b, red_a),
        'mean_nearest_distance_delta_km':   safe_delta(mnd_b, mnd_a),
    }

# batch loop

all_event_summaries    = []
all_region_rows        = []
dashboard_entries       = []

for case in case_studies:
    name= case['name']
    out_folder_event = os.path.join(out_root, name)
    os.makedirs(out_folder_event, exist_ok=True)

    before_df, before_anomaly, before_region, before_tests, before_plots, before_tables = \
        run_event_period(
            name, variable,
            case['lat_min'], case['lat_max'], case['lon_min'], case['lon_max'],
            case['prev_start'], case['prev_end']
        )

    after_df, after_anomaly, after_region, after_tests, after_plots, after_tables = \
        run_event_period(
            name, variable,
            case['lat_min'], case['lat_max'], case['lon_min'], case['lon_max'],
            case['start'], case['end']
        )

    merged = before_df.merge(
        after_df, on=['ID', 'LAT', 'LON', 'NAME'],
        how='outer', suffixes=('_Before', '_After')
    )
    merged.to_csv(
        os.path.join(out_folder_event, 'Before_vs_After_malfunction_summary.csv'), index=False)

    compare_summary = pd.DataFrame([{
        'Event':               name,
        'Before_mean_suspect': before_df['Suspect_%'].mean() if not before_df.empty else np.nan,
        'After_mean_suspect':  after_df['Suspect_%'].mean()  if not after_df.empty  else np.nan,
        'Delta_mean_suspect':  (after_df['Suspect_%'].mean() - before_df['Suspect_%'].mean())
                               if (not after_df.empty and not before_df.empty) else np.nan,
        'Before_flagged_count': int(before_df['Likely_Malfunctioning'].sum()) if not before_df.empty else 0,
        'After_flagged_count':  int(after_df['Likely_Malfunctioning'].sum())  if not after_df.empty  else 0,
    }])
    compare_summary.to_csv(
        os.path.join(out_folder_event, 'before_after_summary.csv'), index=False)
    all_event_summaries.append(compare_summary)

    net_metrics = compute_network_metrics(before_df, after_df)
    pd.DataFrame([net_metrics]).to_csv(
        os.path.join(out_folder_event, 'network_resilience_metrics.csv'), index=False)

    brf_path = os.path.join(before_tables, 'region_suspect_stats.csv')
    arf_path = os.path.join(after_tables,  'region_suspect_stats.csv')

    if os.path.exists(brf_path):
        brf = pd.read_csv(brf_path)
        brf['Event']  = name
        brf['Period'] = 'Before'
        all_region_rows.append(brf)

    if os.path.exists(arf_path):
        arf = pd.read_csv(arf_path)
        arf['Event']  = name
        arf['Period'] = 'After'
        all_region_rows.append(arf)

    dashboard_entries.append({
        'event':           name,
        'folder':          out_folder_event,
        'before_summary':  os.path.join(out_folder_event, 'before_after_summary.csv'),
        'network_metrics': os.path.join(out_folder_event, 'network_resilience_metrics.csv'),
        'plots_folder':    after_plots,
    })

# CSV
print("\nWriting aggregate CSVs...")
if all_event_summaries:
    pd.concat(all_event_summaries, ignore_index=True).to_csv(
        os.path.join(out_root, "CaseStudy_Before_After_Comparison_All.csv"), index=False)

if all_region_rows:
    pd.concat(all_region_rows, ignore_index=True).to_csv(
        os.path.join(out_root, "CaseStudy_Region_Comparison_All.csv"), index=False)

# Plots
print("Generating aggregate plots...")
try:
    event_summ = pd.read_csv(os.path.join(out_root, "CaseStudy_Before_After_Comparison_All.csv"))
    x = np.arange(len(event_summ))
    w = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - w / 2, event_summ['Before_mean_suspect'], width=w, label='Before')
    plt.bar(x + w / 2, event_summ['After_mean_suspect'],  width=w, label='After')
    plt.xticks(x, event_summ['Event'])
    plt.ylabel("Mean % Days Flagged Suspect (vs. neighbours)")
    plt.title("Spatially Suspect Readings Before vs After Storm Events")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "Before_vs_After_Main_Comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(x - w / 2, event_summ['Before_flagged_count'], width=w, label='Before')
    plt.bar(x + w / 2, event_summ['After_flagged_count'],  width=w, label='After')
    plt.xticks(x, event_summ['Event'])
    plt.ylabel("# Stations Flagged Likely_Malfunctioning")
    plt.title("Count of Likely-Malfunctioning Stations Before vs After Storm Events")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "Before_vs_After_Malfunction_Count.png"))
    plt.close()
except Exception as e:
    print(f"  Before/after plot failed: {e}")

try:
    region_all    = pd.read_csv(os.path.join(out_root, "CaseStudy_Region_Comparison_All.csv"))
    region_subset = region_all[region_all["Region"].isin(["Coastal", "Rural"])]
    plt.figure(figsize=(10, 6))
    for event in region_subset["Event"].unique():
        sub = region_subset[
            (region_subset["Event"] == event) & (region_subset["Period"] == "After")]
        plt.scatter(sub["Region"], sub["mean"], label=event, s=80)
    plt.ylabel("Mean % Days Flagged Suspect (After Event)")
    plt.title("Coastal vs Rural Suspect Readings Across Case Studies (After Storms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "Coastal_vs_Rural_Across_Case_Studies.png"))
    plt.close()
except Exception as e:
    print(f"  Coastal/rural plot failed: {e}")

# HTML dashboard
def make_dashboard_html(entries):
    html = [
        "<html><head><meta charset='utf-8'>",
        "<title>Hurricane Station Reliability Dashboard</title>",
        "<style>body{font-family:sans-serif;margin:20px;} img{border:1px solid #ccc;}</style>",
        "</head><body>",
        "<h1>Hurricane Station Reliability — Dashboard</h1>",
        f"<p>Variable: {variable}</p>",
        "<h2>Summary Plots</h2>",
    ]
    if os.path.exists(os.path.join(out_root, "Before_vs_After_Main_Comparison.png")):
        html.append("<h3>Before vs After — Suspect Readings, All Events</h3>")
        html.append("<img src='Before_vs_After_Main_Comparison.png' width=800/>")
    if os.path.exists(os.path.join(out_root, "Before_vs_After_Malfunction_Count.png")):
        html.append("<h3>Before vs After — Stations Flagged Likely_Malfunctioning</h3>")
        html.append("<img src='Before_vs_After_Malfunction_Count.png' width=800/>")
    if os.path.exists(os.path.join(out_root, "Coastal_vs_Rural_Across_Case_Studies.png")):
        html.append("<h3>Coastal vs Rural — After Events</h3>")
        html.append("<img src='Coastal_vs_Rural_Across_Case_Studies.png' width=800/>")
    html.append("<hr/>")

    for e in entries:
        html.append(f"<h2>{e['event']}</h2><ul>")
        for label, path in [
            ("Before/After summary (CSV)", os.path.join(e['folder'], 'before_after_summary.csv')),
            ("Network resilience metrics (CSV)", os.path.join(e['folder'], 'network_resilience_metrics.csv')),
            ("Station Before vs After (CSV)", os.path.join(e['folder'], 'Before_vs_After_malfunction_summary.csv')),
        ]:
            if os.path.exists(path):
                rel = os.path.relpath(path, start=out_root)
                html.append(f"<li><a href='{rel}'>{label}</a></li>")
        html.append("</ul>")

        pf = e['plots_folder']
        if os.path.isdir(pf):
            imgs = sorted(glob.glob(os.path.join(pf, "*.png")))[:6]
            for img in imgs:
                rel = os.path.relpath(img, start=out_root)
                html.append(
                    f"<div style='display:inline-block;margin:5px;'>"
                    f"<img src='{rel}' width=300/><br/>"
                    f"<small>{os.path.basename(img)}</small></div>"
                )
        html.append("<hr/>")

    html.append("</body></html>")
    return "\n".join(html)

print("Writing dashboard.html...")
with open(os.path.join(out_root, "dashboard.html"), 'w') as fh:
    fh.write(make_dashboard_html(dashboard_entries))

print(f"\nDone. Outputs are in the '{out_root}' folder. ")
