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



USE_GEOPANDAS = True
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    USE_GEOPANDAS = False
    print("GeoPandas not available — using fallback methods.")

USE_SEABORN = True
try:
    import seaborn as sns
except Exception:
    USE_SEABORN = False
    print("Seaborn not available — using matplotlib alone.")

USE_SCIPY = True
try:
    from scipy.stats import f_oneway, kruskal, ttest_ind, mannwhitneyu, chi2_contingency, spearmanr, kendalltau, ttest_rel, levene
    from scipy.interpolate import griddata
except Exception:
    USE_SCIPY = False
    print("SciPy not available — some statistical/heatmap features will fallback or be limited.")

# CASE STUDIES

case_studies = [
    {"name":"Sandy","lat_min":39.5,"lat_max":42.5,"lon_min":-75.5,"lon_max":-72.0,
     "start":"2012-10-25","end":"2012-11-05","prev_start":"2011-10-25","prev_end":"2011-11-05"},
    {"name":"Ian","lat_min":26.0,"lat_max":34.0,"lon_min":-84.0,"lon_max":-78.5,
     "start":"2022-09-25","end":"2022-10-05","prev_start":"2021-09-25","prev_end":"2021-10-05"},
    {"name":"Harvey","lat_min":27.0,"lat_max":31.5,"lon_min":-96.5,"lon_max":-91.0,
     "start":"2017-08-23","end":"2017-09-05","prev_start":"2016-08-23","prev_end":"2016-09-05"},
    {"name":"Rita","lat_min":26.0,"lat_max":33.5,"lon_min":-98.5,"lon_max":-87.0,
     "start":"2005-09-18","end":"2005-09-26","prev_start":"2004-09-18","prev_end":"2004-09-26"},
    {"name":"Michael","lat_min":29.0,"lat_max":33.0,"lon_min":-86.5,"lon_max":-83.0,
     "start":"2018-10-07","end":"2018-10-12","prev_start":"2017-10-07","prev_end":"2017-10-12"}
]


# USER VARIABLE

variable = input("Enter variable (TMAX, TMIN, PRCP, TAVG): ").strip().upper()

# FILES 

station_file = 'ghcnd-stations.txt'
inventory_file = 'ghcnd-inventory.txt'
dly_folder = 'dly_files'
os.makedirs(dly_folder, exist_ok=True)
os.makedirs('output', exist_ok=True)

# Check required GHCN files
for req in [station_file, inventory_file]:
    if not os.path.exists(req):
        print(f"Missing required file: {req}\nDownload from https://www.ncei.noaa.gov/pub/data/ghcn/daily/")
        raise SystemExit

# Load inventory + stations

inventory = pd.read_csv(inventory_file, sep=r'\s+', header=None,
                        names=["ID","LAT","LON","ELEMENT","FIRSTYEAR","LASTYEAR"], engine='python')
inventory['ID'] = inventory['ID'].astype(str).str.strip()

station_data = []
with open(station_file, 'r') as fh:
    for line in fh:
        sid = line[0:11].strip()
        lat = float(line[12:20].strip())
        lon = float(line[21:30].strip())
        name = line[41:71].strip()
        station_data.append((sid, lat, lon, name))
station_df = pd.DataFrame(station_data, columns=['ID','LAT','LON','NAME'])
station_df['ID'] = station_df['ID'].astype(str).str.strip()


# Parse .dly function

def parse_dly(filepath, variable, start_date, end_date):
    days_dict = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line[17:21] == variable:
                    year = int(line[11:15])
                    month = int(line[15:17])
                    for i in range(31):
                        try:
                            val_str = line[21+i*8:26+i*8]
                            val = int(val_str[:5])
                            d = date(year, month, i+1)
                            if start_date <= d <= end_date and val != -9999:
                                days_dict[d] = 1
                        except:
                            continue
    except Exception as e:
        print("parse error:", e)
    return days_dict


# GIS utilities & fallback

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

COAST_POINTS = [
    (47.6, -122.3), (37.8, -122.4), (34.0, -118.2), (32.7, -117.1),
    (29.76, -95.37), (30.3, -81.4), (25.8, -80.2), (27.8, -82.6),
    (30.2, -89.2), (40.7, -74.0), (39.3, -76.6), (36.85, -75.98), (43.65, -70.25)
]

CITY_CENTROIDS = [
    ("New York", 40.7128, -74.0060),
    ("Philadelphia", 39.9526, -75.1652),
    ("Baltimore", 39.2904, -76.6122),
    ("Boston", 42.3601, -71.0589),
    ("Miami", 25.7617, -80.1918),
    ("Tampa", 27.9506, -82.4572),
    ("Jacksonville", 30.3322, -81.6557),
    ("Houston", 29.7604, -95.3698),
    ("New Orleans",29.9511, -90.0715),
    ("Tallahassee",30.4383, -84.2807),
    ("Atlanta",33.7490, -84.3880)
]

def download_and_load_naturalearth():
    """Fixed Natural Earth URLs"""
    try:
        
        coast_url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip"
        urban_url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_urban_areas.zip"
        
       
        coast_url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
        urban_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_urban_areas.zip"
        
        tmpdir = "temp_naturalearth"
        os.makedirs(tmpdir, exist_ok=True)
        
        print("Downloading coastline data...")
        r = requests.get(coast_url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(tmpdir)
        coast_shps = [os.path.join(tmpdir,f) for f in os.listdir(tmpdir) if f.endswith('.shp') and 'coastline' in f.lower()]
        if not coast_shps:
            coast_shps = [os.path.join(tmpdir,f) for f in os.listdir(tmpdir) if f.endswith('.shp')]
        gdf_coast = gpd.read_file(coast_shps[0])
        
        print("Downloading urban areas data...")
        r2 = requests.get(urban_url, timeout=60)
        r2.raise_for_status()
        z2 = zipfile.ZipFile(io.BytesIO(r2.content))
        z2.extractall(tmpdir)
        urban_shps = [os.path.join(tmpdir,f) for f in os.listdir(tmpdir) if f.endswith('.shp') and 'urban' in f.lower()]
        if len(urban_shps) == 0:
            urban_shps = [os.path.join(tmpdir,f) for f in os.listdir(tmpdir) if f.endswith('.shp')]
        gdf_urban = gpd.read_file(urban_shps[0])
        
        print("Natural Earth data loaded successfully.")
        return gdf_coast, gdf_urban
    except Exception as e:
        print("Natural Earth download/load failed:", e)
        return None, None

def compute_distances_and_regions(stations_df, output_folder, coastal_km_thresh=50):
    st = stations_df.copy().reset_index(drop=True)
    if USE_GEOPANDAS:
        try:
            gdf_coast, gdf_urban = download_and_load_naturalearth()
            if gdf_coast is not None:
                gdf_coast_proj = gdf_coast.to_crs(epsg=3857)
                coast_union = gdf_coast_proj.unary_union
                pts = gpd.GeoDataFrame(st, geometry=[Point(xy) for xy in zip(st['LON'], st['LAT'])], crs="EPSG:4326")
                pts_proj = pts.to_crs(epsg=3857)
                dists_m = pts_proj.geometry.apply(lambda p: p.distance(coast_union))
                distances = dists_m.values / 1000.0
                if gdf_urban is not None:
                    gdf_urban_proj = gdf_urban.to_crs(epsg=3857)
                    pts_proj['in_urban'] = pts_proj.within(gdf_urban_proj.unary_union)
                    in_urban = pts_proj['in_urban'].values
                else:
                    in_urban = np.array([False]*len(st))
                regions = []
                for i, d in enumerate(distances):
                    if d <= coastal_km_thresh:
                        regions.append('Coastal')
                    else:
                        if in_urban[i]:
                            regions.append('Inland')
                        else:
                            regions.append('Rural')
                st['Distance_to_Coast_km'] = distances
                st['Region'] = regions
                return st
        except Exception as e:
            print("GeoPandas method failed, falling back:", e)
    
    # Fallback method
    print("Using fallback distance calculation...")
    dist_to_coast = []
    is_urban = []
    for idx, row in st.iterrows():
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
        else:
            if is_urban[i]:
                regions.append('Inland')
            else:
                regions.append('Rural')
    st['Region'] = regions
    pd.DataFrame({'ID':st['ID'],'Distance_to_Coast_km':st['Distance_to_Coast_km'],'Region':st['Region']}) \
      .to_csv(os.path.join(output_folder, 'region_assignment_fallback.csv'), index=False)
    return st


# run_event_period
##
def run_event_period(event_name, variable, lat_min, lat_max, lon_min, lon_max, start_str, end_str):
    output_folder = os.path.join('output', event_name, f"{start_str}_to_{end_str}")
    os.makedirs(output_folder, exist_ok=True)
    plots_folder = os.path.join(output_folder, "plots"); os.makedirs(plots_folder, exist_ok=True)
    tables_folder = os.path.join(output_folder, "tables"); os.makedirs(tables_folder, exist_ok=True)

    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_str, "%Y-%m-%d").date()

    inv_filtered = inventory[(inventory['ELEMENT']==variable) &
                             (inventory['FIRSTYEAR'] <= start_date.year) &
                             (inventory['LASTYEAR'] >= end_date.year)]

    stations = station_df[(station_df['LAT'] >= lat_min) & (station_df['LAT'] <= lat_max) &
                          (station_df['LON'] >= lon_min) & (station_df['LON'] <= lon_max)]
    stations = pd.merge(stations, inv_filtered[['ID']], on='ID', how='inner')

    base_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all"
    for sid in stations['ID'].unique():
        dest = os.path.join(dly_folder, f"{sid}.dly")
        if not os.path.exists(dest):
            try:
                r = requests.get(f"{base_url}/{sid}.dly", timeout=30)
                if r.status_code == 200:
                    with open(dest, 'wb') as fh:
                        fh.write(r.content)
            except Exception:
                pass

    date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    total = np.zeros(len(date_list)); valid = np.zeros(len(date_list))
    missing_station_ids = [[] for _ in range(len(date_list))]

    for sid in stations['ID'].unique():
        path = os.path.join(dly_folder, f"{sid}.dly")
        if os.path.exists(path):
            vals = parse_dly(path, variable, start_date, end_date)
            for i, d in enumerate(date_list):
                total[i] += 1
                if d in vals:
                    valid[i] += 1
                else:
                    missing_station_ids[i].append(sid)

    summary = pd.DataFrame({
        'Date': [d.strftime("%Y-%m-%d") for d in date_list],
        'Stations Reporting': valid.astype(int),
        'Stations Missing': (total - valid).astype(int),
        'Total Stations': total.astype(int),
        '% Missing': np.round((1 - valid / total) * 100, 1),
        'Missing Station IDs': [','.join(ids) if ids else '' for ids in missing_station_ids]
    })
    summary.to_csv(os.path.join(output_folder, 'missing_summary.csv'), index=False)

    records = []
    for sid in stations['ID'].unique():
        path = os.path.join(dly_folder, f"{sid}.dly")
        if os.path.exists(path):
            vals = parse_dly(path, variable, start_date, end_date)
            total_days = len(date_list)
            missing_days = sum(1 for d in date_list if d not in vals)
            info = stations[stations['ID'] == sid].iloc[0]
            missing_pct = round((missing_days / total_days) * 100, 2)
            records.append({'ID': sid, 'LAT': info['LAT'], 'LON': info['LON'], 'NAME': info['NAME'],
                            'Total Days': total_days, 'Missing Days': missing_days, 'Missing %': missing_pct})

    station_missing_df = pd.DataFrame(records)
    station_missing_df.to_csv(os.path.join(output_folder, 'stations_with_missing_data.csv'), index=False)

    region_df = compute_distances_and_regions(stations[['ID','LAT','LON','NAME']], output_folder, coastal_km_thresh=50)
    station_missing_df = station_missing_df.merge(region_df[['ID','Distance_to_Coast_km','Region']], on='ID', how='left')
    station_missing_df.to_csv(os.path.join(output_folder, 'stations_with_missing_data_region.csv'), index=False)

    # Stats
    out_lines = []
    out_lines.append(f"Event: {event_name} {start_str} to {end_str}")
    out_lines.append(f"Stations considered: {len(station_missing_df)}")
    out_lines.append("")

    if len(station_missing_df) >= 3:
        corr_lat = station_missing_df['LAT'].corr(station_missing_df['Missing %'])
        corr_lon = station_missing_df['LON'].corr(station_missing_df['Missing %'])
        out_lines.append(f"Pearson correlation: LAT vs Missing% = {corr_lat:.3f}; LON vs Missing% = {corr_lon:.3f}")
        if USE_SCIPY:
            rho, p_rho = spearmanr(station_missing_df['Distance_to_Coast_km'], station_missing_df['Missing %'])
            tau, p_tau = kendalltau(station_missing_df['Distance_to_Coast_km'], station_missing_df['Missing %'])
            out_lines.append(f"Spearman (distance->missing) rho={rho:.3f}, p={p_rho:.3f}")
            out_lines.append(f"Kendall tau={tau:.3f}, p={p_tau:.3f}")
    else:
        out_lines.append("Not enough stations for correlations.")

    # Region stats table
    region_stats = station_missing_df.groupby('Region')['Missing %'].agg(['mean','std','count']).reset_index()
    region_stats.to_csv(os.path.join(tables_folder, 'region_missing_stats.csv'), index=False)

    # Statistical tests (ANOVA, Kruskal, pairwise, chi-square)
    test_results = []
    groups = {r: station_missing_df[station_missing_df['Region']==r]['Missing %'].dropna().values
              for r in station_missing_df['Region'].unique()}

    if USE_SCIPY and len(groups) >= 2 and all(len(v) >= 2 for v in groups.values()):
        try:
            anova_stat, anova_p = f_oneway(*groups.values())
            test_results.append({'test':'ANOVA','stat':anova_stat,'p':anova_p})
        except Exception:
            pass
        try:
            kw_stat, kw_p = kruskal(*groups.values())
            test_results.append({'test':'Kruskal','stat':kw_stat,'p':kw_p})
        except Exception:
            pass

    # Pairwise Coastal vs Rural/Inland t-test + MWU
    if 'Coastal' in groups and 'Rural' in groups:
        g_coastal = groups['Coastal']; g_rural = groups['Rural']
        if USE_SCIPY and len(g_coastal)>=2 and len(g_rural)>=2:
            try:
                t_stat, t_p = ttest_ind(g_coastal, g_rural, equal_var=False)
                test_results.append({'test':'Ttest_Coastal_vs_Rural','stat':t_stat,'p':t_p})
                mw_stat, mw_p = mannwhitneyu(g_coastal, g_rural, alternative='two-sided')
                test_results.append({'test':'MannWhitney_Coastal_vs_Rural','stat':mw_stat,'p':mw_p})
            except Exception as e:
                print(f"Pairwise test Coastal vs Rural failed: {e}")
    
    if 'Coastal' in groups and 'Inland' in groups:
        g_coastal = groups['Coastal']; g_inland = groups['Inland']
        if USE_SCIPY and len(g_coastal)>=2 and len(g_inland)>=2:
            try:
                t_stat2, t_p2 = ttest_ind(g_coastal, g_inland, equal_var=False)
                test_results.append({'test':'Ttest_Coastal_vs_Inland','stat':t_stat2,'p':t_p2})
                mw_stat2, mw_p2 = mannwhitneyu(g_coastal, g_inland, alternative='two-sided')
                test_results.append({'test':'MannWhitney_Coastal_vs_Inland','stat':mw_stat2,'p':mw_p2})
            except Exception as e:
                print(f"Pairwise test Coastal vs Inland failed: {e}")

    # Chi-square contingency - FIXED: added minimum cell count check
    try:
        median_missing = station_missing_df['Missing %'].median()
        station_missing_df['MissingCat'] = np.where(station_missing_df['Missing %'] > median_missing, 'High', 'Low')
        contingency = pd.crosstab(station_missing_df['Region'], station_missing_df['MissingCat'])
        # Check minimum expected frequency (should be >= 5)
        if USE_SCIPY and contingency.shape[0] > 1 and contingency.shape[1] > 1 and contingency.min().min() >= 5:
            chi2, p_chi, dof, exp = chi2_contingency(contingency)
            test_results.append({'test':'ChiSquare_region_vs_missingcat','stat':chi2,'p':p_chi})
    except Exception as e:
        print(f"Chi-square test failed: {e}")

    tests_df = pd.DataFrame(test_results)
    tests_df.to_csv(os.path.join(tables_folder, 'statistical_tests_results.csv'), index=False)

    # PLOTS: box/violin/scatter/heatmap
    if not station_missing_df.empty:
        if USE_SEABORN:
            sns.set(style="whitegrid")
            plt.figure(figsize=(8,6))
            sns.boxplot(data=station_missing_df, x="Region", y="Missing %")
            plt.title(f"{event_name}: Missing % by Region")
            plt.tight_layout(); plt.savefig(os.path.join(plots_folder, 'boxplot_missing_by_region.png')); plt.close()

            plt.figure(figsize=(8,6))
            sns.violinplot(data=station_missing_df, x="Region", y="Missing %")
            plt.title(f"{event_name}: Missing % distribution by Region")
            plt.tight_layout(); plt.savefig(os.path.join(plots_folder, 'violin_missing_by_region.png')); plt.close()
        else:
            plt.figure(figsize=(8,6))
            station_missing_df.boxplot(column='Missing %', by='Region')
            plt.title(f"{event_name}: Missing % by Region"); plt.suptitle("")
            plt.tight_layout(); plt.savefig(os.path.join(plots_folder, 'boxplot_missing_by_region.png')); plt.close()

        # scatter lat/lon heatmap / colored points
        plt.figure(figsize=(8,6))
        sc = plt.scatter(station_missing_df['LON'], station_missing_df['LAT'], c=station_missing_df['Missing %'], s=40, cmap='Reds', edgecolor='k', linewidth=0.2)
        plt.colorbar(sc, label='% Missing')
        plt.title(f"{event_name}: Spatial Missing % (points colored by % Missing)")
        plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, 'spatial_scatter_missing.png')); plt.close()

        # heatmap interpolation if scipy available
        try:
            if USE_SCIPY and len(station_missing_df) >= 4:  # Need at least 4 points for interpolation
                xi = np.linspace(station_missing_df['LON'].min(), station_missing_df['LON'].max(), 200)
                yi = np.linspace(station_missing_df['LAT'].min(), station_missing_df['LAT'].max(), 200)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = griddata((station_missing_df['LON'], station_missing_df['LAT']), station_missing_df['Missing %'], (Xi, Yi), method='linear')
                plt.figure(figsize=(8,6))
                plt.contourf(Xi, Yi, Zi, levels=14, cmap='Reds', alpha=0.8)
                plt.scatter(station_missing_df['LON'], station_missing_df['LAT'], c=station_missing_df['Missing %'], s=20, cmap='Reds', edgecolor='k', linewidth=0.2)
                plt.colorbar(label='% Missing')
                plt.title(f"{event_name}: Interpolated Heatmap of % Missing")
                plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.tight_layout()
                plt.savefig(os.path.join(plots_folder, 'heatmap_interpolated_missing.png')); plt.close()
        except Exception as e:
            print(f"Heatmap interpolation failed: {e}")

    # Save interpretation text
    with open(os.path.join(output_folder, 'interpretation_summary.txt'), 'w') as fh:
        fh.write("\n".join(out_lines))
    
    return station_missing_df, summary, region_stats, tests_df, plots_folder, tables_folder

def compute_network_metrics(before_df, after_df):
    # define failed if Missing % >= 50
    thr = 50.0
    def failure_mask(df):
        if df is None or df.empty: return pd.DataFrame(columns=['ID','LAT','LON','Failed'])
        m = df.copy()
        m['Failed'] = m['Missing %'] >= thr
        return m[['ID','LAT','LON','Failed']]
    b = failure_mask(before_df); a = failure_mask(after_df)
    
    # coverage loss: proportion failed
    def coverage(df):
        if df is None or df.empty: return np.nan
        return df['Failed'].sum() / len(df)
    coverage_before = coverage(b); coverage_after = coverage(a)
    
    # redundancy ratio = total / (# failed + 1e-9)
    def redundancy(df):
        if df is None or df.empty: return np.nan
        total = len(df)
        failed = df['Failed'].sum()
        return total / (failed + 1e-9)
    redundancy_before = redundancy(b); redundancy_after = redundancy(a)
    
    # mean nearest-working distance: for each failed station, distance to nearest non-failed station
    def mean_nearest_distance(df):
        if df is None or df.empty: return np.nan
        coords = df[['ID','LAT','LON','Failed']].copy()
        failed = coords[coords['Failed']]
        working = coords[~coords['Failed']]
        if failed.empty or working.empty: return np.nan
        dlist = []
        for _, f in failed.iterrows():
            latf, lonf = f['LAT'], f['LON']
            dmin = working.apply(lambda r: haversine_km(latf, lonf, r['LAT'], r['LON']), axis=1).min()
            dlist.append(dmin)
        return np.mean(dlist) if dlist else np.nan
    mean_nd_before = mean_nearest_distance(b); mean_nd_after = mean_nearest_distance(a)
    
    # return dict
    return {
        'coverage_before': coverage_before,
        'coverage_after': coverage_after,
        'redundancy_before': redundancy_before,
        'redundancy_after': redundancy_after,
        'mean_nearest_distance_before_km': mean_nd_before,
        'mean_nearest_distance_after_km': mean_nd_after,
        'coverage_delta': (coverage_after - coverage_before) if (not np.isnan(coverage_before) and not np.isnan(coverage_after)) else np.nan,
        'redundancy_delta': (redundancy_after - redundancy_before) if (not np.isnan(redundancy_before) and not np.isnan(redundancy_after)) else np.nan,
        'mean_nearest_distance_delta_km': (mean_nd_after - mean_nd_before) if (not np.isnan(mean_nd_before) and not np.isnan(mean_nd_after)) else np.nan
    }

all_event_summaries = []
all_region_rows = []
dashboard_entries = []

for case in case_studies:
    name = case['name']
    out_folder_event = os.path.join('output', name)
    os.makedirs(out_folder_event, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {name} - Before Period")
    print(f"{'='*60}")
    before_df, before_summary, before_region, before_tests, before_plots, before_tables = run_event_period(
        name, variable, case['lat_min'], case['lat_max'], case['lon_min'], case['lon_max'], case['prev_start'], case['prev_end'])

    print(f"\n{'='*60}")
    print(f"Processing {name} - After Period (Hurricane)")
    print(f"{'='*60}")
    after_df, after_summary, after_region, after_tests, after_plots, after_tables = run_event_period(
        name, variable, case['lat_min'], case['lat_max'], case['lon_min'], case['lon_max'], case['start'], case['end'])

    merged = before_df.merge(after_df, on=['ID','LAT','LON','NAME'], how='outer', suffixes=('_Before','_After'))
    merged.to_csv(os.path.join(out_folder_event, 'Before_vs_After_station_missing.csv'), index=False)

    compare_summary = pd.DataFrame([{
        'Event': name,
        'Before_mean_missing': before_df['Missing %'].mean() if not before_df.empty else np.nan,
        'After_mean_missing': after_df['Missing %'].mean() if not after_df.empty else np.nan,
        'Delta_mean_missing': (after_df['Missing %'].mean() - before_df['Missing %'].mean()) if (not after_df.empty and not before_df.empty) else np.nan
    }])
    compare_summary.to_csv(os.path.join(out_folder_event, 'before_after_summary.csv'), index=False)
    all_event_summaries.append(compare_summary)

    # compute network metrics
    net_metrics = compute_network_metrics(before_df, after_df)
    net_df = pd.DataFrame([net_metrics])
    net_df.to_csv(os.path.join(out_folder_event, 'network_resilience_metrics.csv'), index=False)

    # Collect region stats for aggregate - FIXED paths
    before_region_path = os.path.join(out_folder_event, f"{case['prev_start']}_to_{case['prev_end']}", "tables", "region_missing_stats.csv")
    after_region_path = os.path.join(out_folder_event, f"{case['start']}_to_{case['end']}", "tables", "region_missing_stats.csv")
    
    if os.path.exists(before_region_path):
        brf = pd.read_csv(before_region_path)
        brf['Event'] = name
        brf['Period'] = 'Before'
        all_region_rows.append(brf)
    
    if os.path.exists(after_region_path):
        arf = pd.read_csv(after_region_path)
        arf['Event'] = name
        arf['Period'] = 'After'
        all_region_rows.append(arf)

    # dashboard info - FIXED: store actual case for this event
    dashboard_entries.append({
        'event': name,
        'case': case,  # Store the full case dict
        'folder': out_folder_event,
        'before_summary': os.path.join(out_folder_event, 'before_after_summary.csv'),
        'network_metrics': os.path.join(out_folder_event, 'network_resilience_metrics.csv'),
        'before_plots': os.path.join(out_folder_event, f"{case['prev_start']}_to_{case['prev_end']}", "plots"),
        'after_plots': os.path.join(out_folder_event, f"{case['start']}_to_{case['end']}", "plots")
    })

# aggregate
if all_event_summaries:
    event_summaries = pd.concat(all_event_summaries, ignore_index=True)
    event_summaries.to_csv("output/CaseStudy_Before_After_Comparison_All.csv", index=False)
if all_region_rows:
    region_all = pd.concat(all_region_rows, ignore_index=True)
    region_all.to_csv("output/CaseStudy_Region_Comparison_All.csv", index=False)

try:
    event_summ = pd.read_csv("output/CaseStudy_Before_After_Comparison_All.csv")
    plt.figure(figsize=(10,6))
    x = np.arange(len(event_summ))
    w = 0.35
    plt.bar(x - w/2, event_summ['Before_mean_missing'], width=w, label='Before', alpha=0.8)
    plt.bar(x + w/2, event_summ['After_mean_missing'], width=w, label='After (Hurricane)', alpha=0.8)
    plt.xticks(x, event_summ['Event'])
    plt.ylabel("Mean % Missing Data")
    plt.title(f"Station Data Loss Before vs After Storm Events - {variable}")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/Before_vs_After_Main_Comparison.png", dpi=150)
    plt.close()
    print("\nSaved aggregate comparison plot: output/Before_vs_After_Main_Comparison.png")
except Exception as e:
    print(f"Failed to create main comparison plot: {e}")

# Coastal vs Rural across case studies
try:
    region_all = pd.read_csv("output/CaseStudy_Region_Comparison_All.csv")
    region_subset = region_all[region_all["Region"].isin(["Coastal","Rural"])]
    plt.figure(figsize=(10,6))
    for event in region_subset["Event"].unique():
        sub = region_subset[(region_subset["Event"] == event) & (region_subset["Period"] == "After")]
        if not sub.empty:
            plt.scatter(sub["Region"], sub["mean"], label=event, s=100, alpha=0.7)
    plt.ylabel("Mean % Missing Data (After Event)")
    plt.xlabel("Region")
    plt.title(f"Coastal vs Rural Data Loss Across Case Studies (After Hurricanes) - {variable}")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/Coastal_vs_Rural_Across_Case_Studies.png", dpi=150)
    plt.close()
    print("Saved coastal vs rural plot: output/Coastal_vs_Rural_Across_Case_Studies.png")
except Exception as e:
    print(f"Failed to create coastal vs rural plot: {e}")

def make_dashboard_html(entries):
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Hurricane Station Reliability Dashboard</title>"]
    html.append("<style>")
    html.append("body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
    html.append("h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }")
    html.append("h2 { color: #34495e; margin-top: 30px; }")
    html.append("h3 { color: #7f8c8d; }")
    html.append(".summary-section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
    html.append(".event-section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
    html.append("ul { line-height: 1.8; }")
    html.append(".plot-gallery { display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }")
    html.append(".plot-item { text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background: #fafafa; }")
    html.append(".plot-item img { max-width: 350px; height: auto; border-radius: 4px; }")
    html.append(".plot-item small { display: block; margin-top: 5px; color: #666; }")
    html.append("</style>")
    html.append("</head><body>")
    
    html.append("<h1>🌪️ Hurricane Station Reliability Dashboard</h1>")
    html.append(f"<div class='summary-section'><p><strong>Variable:</strong> {variable}</p>")
    html.append(f"<p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>")
    
    html.append("<div class='summary-section'>")
    html.append("<h2>📊 Summary Plots</h2>")
    if os.path.exists("output/Before_vs_After_Main_Comparison.png"):
        html.append("<h3>Before vs After — All Events</h3>")
        html.append("<img src='Before_vs_After_Main_Comparison.png' width='800' style='border: 1px solid #ddd; border-radius: 4px;'/>")
    if os.path.exists("output/Coastal_vs_Rural_Across_Case_Studies.png"):
        html.append("<h3>Coastal vs Rural — After Events</h3>")
        html.append("<img src='Coastal_vs_Rural_Across_Case_Studies.png' width='800' style='border: 1px solid #ddd; border-radius: 4px;'/>")
    html.append("</div>")
    
    html.append("<hr/>")
    
    for e in entries:
        html.append(f"<div class='event-section'>")
        html.append(f"<h2>🌀 {e['event']}</h2>")
        
   
        before_path = e['before_summary']
        net_path = e['network_metrics']
        merged_path = os.path.join(e['folder'], 'Before_vs_After_station_missing.csv')
        
        html.append("<h3>📁 Data Files</h3>")
        html.append("<ul>")
        if os.path.exists(before_path):
            rel_path = os.path.relpath(before_path, start='output')
            html.append(f"<li><a href='{rel_path}'>Before/After Summary (CSV)</a></li>")
        if os.path.exists(net_path):
            rel_path = os.path.relpath(net_path, start='output')
            html.append(f"<li><a href='{rel_path}'>Network Resilience Metrics (CSV)</a></li>")
        if os.path.exists(merged_path):
            rel_path = os.path.relpath(merged_path, start='output')
            html.append(f"<li><a href='{rel_path}'>Station Before vs After (CSV)</a></li>")
        html.append("</ul>")
        
      
        if os.path.exists(net_path):
            try:
                net_data = pd.read_csv(net_path)
                html.append("<h3>📈 Network Resilience Metrics</h3>")
                html.append("<table style='border-collapse: collapse; width: 100%;'>")
                html.append("<tr style='background: #3498db; color: white;'>")
                html.append("<th style='padding: 10px; text-align: left;'>Metric</th>")
                html.append("<th style='padding: 10px; text-align: right;'>Before</th>")
                html.append("<th style='padding: 10px; text-align: right;'>After</th>")
                html.append("<th style='padding: 10px; text-align: right;'>Change</th>")
                html.append("</tr>")
                
                if not net_data.empty:
                    row = net_data.iloc[0]
                    html.append(f"<tr style='background: #ecf0f1;'><td style='padding: 8px;'>Coverage (% Failed)</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('coverage_before', 0):.3f}</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('coverage_after', 0):.3f}</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('coverage_delta', 0):.3f}</td></tr>")
                    
                    html.append(f"<tr><td style='padding: 8px;'>Redundancy Ratio</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('redundancy_before', 0):.2f}</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('redundancy_after', 0):.2f}</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('redundancy_delta', 0):.2f}</td></tr>")
                    
                    html.append(f"<tr style='background: #ecf0f1;'><td style='padding: 8px;'>Mean Distance to Working Station (km)</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('mean_nearest_distance_before_km', 0):.1f}</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('mean_nearest_distance_after_km', 0):.1f}</td>")
                    html.append(f"<td style='padding: 8px; text-align: right;'>{row.get('mean_nearest_distance_delta_km', 0):.1f}</td></tr>")
                
                html.append("</table>")
            except Exception as ex:
                html.append(f"<p style='color: red;'>Error loading network metrics: {ex}</p>")
        
 
        html.append("<h3>📷 Visualizations</h3>")
        html.append("<div class='plot-gallery'>")
        

        if 'after_plots' in e and os.path.exists(e['after_plots']):
            imgs = sorted(glob.glob(os.path.join(e['after_plots'], "*.png")))
            for img in imgs[:8]:  # Show up to 8 plots
                rel = os.path.relpath(img, start='output')
                basename = os.path.basename(img)
                html.append(f"<div class='plot-item'>")
                html.append(f"<img src='{rel}' alt='{basename}'/>")
                html.append(f"<small>After: {basename.replace('_', ' ').replace('.png', '')}</small>")
                html.append(f"</div>")
        
        html.append("</div>")
        html.append("</div><hr/>")
    
    html.append("</body></html>")
    return "\n".join(html)

print("\n" + "="*60)
print("Generating HTML Dashboard...")
print("="*60)
dashboard_html = make_dashboard_html(dashboard_entries)
dashboard_path = os.path.join("output", "dashboard.html")
with open(dashboard_path, 'w', encoding='utf-8') as fh:
    fh.write(dashboard_html)

print(f"\n Dashboard created: {dashboard_path}")
print(f" All outputs saved in 'output' folder")
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nTo view results:")
print(f"1. Open {dashboard_path} in your web browser")
print(f"2. Review CSV files in output/ folder")
print(f"3. Check individual event folders for detailed plots and statistics")
print("\n" + "="*60)

import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import os

def compare_before_after(event_name, before_csv, during_csv):
    """
    Takes two CSVs generated by your own script:
    - stations_with_missing_data_ BEFORE event
    - stations_with_missing_data_ DURING event
    and performs:
        • Mean comparison
        • Paired t-test
        • Delta (%) reliability change
        • Exports comparison CSV
        • Creates comparison plot
    """

    before = pd.read_csv(before_csv)
    during = pd.read_csv(during_csv)

    # Keep only stations that appear in BOTH periods
    merged = before.merge(
        during,
        on="STATION",
        suffixes=("_before", "_during")
    )

    # Calculate difference
    merged["delta_missing"] = (
        merged["missing_pct_during"] - merged["missing_pct_before"]
    )

    # Stats
    mean_before = merged["missing_pct_before"].mean()
    mean_during = merged["missing_pct_during"].mean()
    mean_change = merged["delta_missing"].mean()

    t_stat, p_val = ttest_rel(
        merged["missing_pct_before"],
        merged["missing_pct_during"]
    )

    # Export results table
    out_df = pd.DataFrame({
        "Mean Missing % (Before)": [mean_before],
        "Mean Missing % (During)": [mean_during],
        "Average Change (%)": [mean_change],
        "t-statistic": [t_stat],
        "p-value": [p_val],
        "Interpretation": [
            "Significant difference" if p_val < 0.05 else "No statistically significant difference"
        ]
    })

    out_dir = f"output/{event_name}/comparisons"
    os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(f"{out_dir}/{event_name}_before_after_stats.csv", index=False)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(["Before", "During"], [mean_before, mean_during])
    plt.title(f"Weather Station Reliability Before vs During {event_name}")
    plt.ylabel("Mean % Missing Data")
    plt.savefig(f"{out_dir}/{event_name}_before_vs_during_plot.png", dpi=300)
    plt.close()

    print("\n=== BEFORE vs DURING COMPARISON COMPLETE ===")
    print(out_df)
    print(f"Files saved in: {out_dir}")
    print("===========================================\n")
