import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained
import numpy as np
import io
import folium
from streamlit_folium import st_folium
import math
import warnings
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# ==========================================
# 0. CẤU HÌNH CHUNG & CSS
# ==========================================
st.set_page_config(layout="wide", page_title="RTM Route Planner", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container { 
            padding-top: 2rem; 
            padding-bottom: 2rem; 
            padding-left: 5rem; 
            padding-right: 5rem;
            max_width: 1400px;
            margin: auto;
        }
        iframe { width: 100% !important; }
        
        div[data-testid="stFormSubmitButton"] > button {
            width: auto !important;
        }
        
        div[data-testid="column"]:nth-of-type(3) button {
            float: right;
        }

        /* CSS cho Tiêu đề Đỏ sẫm */
        .main-title {
            color: #8B0000 !important; /* Dark Red */
            font-weight: bold !important;
            font-size: 2.5rem !important;
            margin-bottom: 1rem !important;
            text-align: center;
        }

        /* CSS cho Heatmap */
        [data-testid="stDataFrame"] td {
            text-align: center !important;
            font-size: 11px !important;
            padding: 0px !important;
            white-space: nowrap !important;
        }
        [data-testid="stDataFrame"] th {
            text-align: center !important;
            font-size: 11px !important;
            padding: 2px !important;
        }
        [data-testid="stDataFrame"] th button { display: none !important; }
        
        /* CSS cho Alert Căn giữa */
        .warning-box {
            background-color: #fffae5;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #ffeeba;
            margin-bottom: 10px;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

def render_main_title():
    st.markdown('<h1 class="main-title">Công cụ xếp tuyến - RTM Route Planner</h1>', unsafe_allow_html=True)

# CẤU HÌNH BẢN ĐỒ ESRI
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Tiles &copy; Esri" # Đã thu gọn tối đa dòng chú thích
# ==========================================
# 1. QUẢN LÝ STATE
# ==========================================
# Global State
if 'global_state' not in st.session_state:
    st.session_state.global_state = {
        'has_started': False,
        'step': 'welcome', 
        'config': {'is_tp': False, 'is_vp': False, 'is_integrated': False, 'tp_mode': "Chế độ 1"}
    }

# State Code 1 (Territory Planner)
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'df' not in st.session_state: st.session_state.df = None
if 'df_edited' not in st.session_state: st.session_state.df_edited = None
if 'col_mapping' not in st.session_state: st.session_state.col_mapping = {}
if 'mapping_confirmed' not in st.session_state: st.session_state.mapping_confirmed = False
if 'time_matrix' not in st.session_state: 
    st.session_state.time_matrix = {
        'MT': 20.0, 'Cooler': 18.0, 'Gold': 15.0, 'Silver': 10.0, 
        'Bronze': 5.0, 'Mặc định/Trống': 10.0
    }
if 'upload_msg' not in st.session_state: st.session_state.upload_msg = None
if 'upload_msg_type' not in st.session_state: st.session_state.upload_msg_type = "success"
if 'v1_df_edited' not in st.session_state: st.session_state.v1_df_edited = None
if 'v1_report' not in st.session_state: st.session_state.v1_report = None
if 'v1_df_original' not in st.session_state: st.session_state.v1_df_original = None
if 'v1_map_snapshot' not in st.session_state: st.session_state.v1_map_snapshot = None
if 'v2_df_edited' not in st.session_state: st.session_state.v2_df_edited = None
if 'v2_report' not in st.session_state: st.session_state.v2_report = None
if 'v2_df_original' not in st.session_state: st.session_state.v2_df_original = None
if 'v2_map_snapshot' not in st.session_state: st.session_state.v2_map_snapshot = None
if 'map_version' not in st.session_state: st.session_state.map_version = 0
if 'col_widths' not in st.session_state: st.session_state.col_widths = {}
if 'last_mode' not in st.session_state: st.session_state.last_mode = "Chế độ 1"

# State Code 2 (Visit Planner)
if 'stage' not in st.session_state: st.session_state.stage = '1_upload'
if 'df_cust' not in st.session_state: st.session_state.df_cust = None
if 'df_dist' not in st.session_state: st.session_state.df_dist = None
if 'df_editing' not in st.session_state: st.session_state.df_editing = None 
if 'df_final' not in st.session_state: st.session_state.df_final = None
if 'map_clicked_code' not in st.session_state: st.session_state.map_clicked_code = None
if 'editor_filter_mode' not in st.session_state: st.session_state.editor_filter_mode = 'all'
if 'col_map_main' not in st.session_state: st.session_state.col_map_main = {}
if 'has_changes' not in st.session_state: st.session_state.has_changes = False
if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False
if 'depot_coords' not in st.session_state: st.session_state.depot_coords = None
if 'vp_msg' not in st.session_state: st.session_state.vp_msg = None
if 'vp_msg_type' not in st.session_state: st.session_state.vp_msg_type = None
# --- Interaction States ---
if 'editor_filter_key' not in st.session_state: st.session_state.editor_filter_key = None
if 'show_download_options' not in st.session_state: st.session_state.show_download_options = False
if 'tp_confirm_clear' not in st.session_state: st.session_state.tp_confirm_clear = False
if 'vp_confirm_clear' not in st.session_state: st.session_state.vp_confirm_clear = False

# ==========================================
# 2. LOGIC FUNCTIONS
# ==========================================

# --- CODE 1 HELPER ---
WORKING_DAYS = 22

@st.cache_data
def load_excel_file(file):
    return pd.read_excel(file, dtype=str)

def run_territory_planning_v1(df, lat_col, lon_col, n_clusters, min_size, max_size, n_init=50):
    df_run = df.copy()
    df_run[lat_col] = pd.to_numeric(df_run[lat_col], errors='coerce')
    df_run[lon_col] = pd.to_numeric(df_run[lon_col], errors='coerce')
    df_run = df_run.dropna(subset=[lat_col, lon_col])
    
    coords = df_run[[lat_col, lon_col]]
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    if min_size * n_clusters > len(df): return None, "Lỗi: Số lượng tối thiểu quá lớn."
    if max_size * n_clusters < len(df): return None, "Lỗi: Số lượng tối đa quá nhỏ."

    progress_text = "Đang xử lý..."
    my_bar = st.progress(0, text=progress_text)
    best_clf = None
    best_inertia = float('inf')

    try:
        for i in range(n_init):
            clf = KMeansConstrained(
                n_clusters=n_clusters, size_min=min_size, size_max=max_size, 
                random_state=42 + i, n_init=1
            )
            clf.fit(coords_scaled)
            if clf.inertia_ < best_inertia:
                best_inertia = clf.inertia_
                best_clf = clf
            percent = int((i + 1) / n_init * 100)
            my_bar.progress((i + 1) / n_init, text=f"Đang xử lý... {percent}%")
            
        my_bar.empty()
        df_run['territory_id'] = best_clf.labels_ + 1
        
        stats = df_run['territory_id'].value_counts().sort_index().reset_index()
        stats.columns = ['Tuyến (RouteID)', 'Số lượng KH']
        
        return df_run, stats
    except Exception as e:
        my_bar.empty()
        return None, str(e)

def run_territory_planning_v2(df, lat_col, lon_col, freq_col, type_col, time_matrix, n_clusters, min_capacity_total, max_capacity_total):
    df_run = df.copy()
    df_run[lat_col] = pd.to_numeric(df_run[lat_col], errors='coerce')
    df_run[lon_col] = pd.to_numeric(df_run[lon_col], errors='coerce')
    df_run = df_run.dropna(subset=[lat_col, lon_col])

    def calc_load(row):
        try: freq = float(row[freq_col])
        except: freq = 1.0
        c_type = str(row[type_col]).strip()
        key = c_type if c_type in time_matrix else 'Mặc định/Trống'
        time_val = time_matrix.get(key, 10.0)
        return freq * time_val

    df_run['workload_min'] = df_run.apply(calc_load, axis=1)
    total_minutes = df_run['workload_min'].sum()
    
    TARGET_POINTS = 50000 
    raw_quantum = total_minutes / TARGET_POINTS
    QUANTUM = max(1, math.ceil(raw_quantum)) 
    
    df_run['weight_points'] = np.ceil(df_run['workload_min'] / QUANTUM).astype(int)
    
    df_exploded = df_run.loc[df_run.index.repeat(df_run['weight_points'])].copy()
    df_exploded['original_index'] = df_exploded.index
    df_exploded = df_exploded.reset_index(drop=True)
    
    size_min = int(min_capacity_total / QUANTUM)
    size_max = int(max_capacity_total / QUANTUM)
    
    scaler = StandardScaler()
    coords = df_exploded[[lat_col, lon_col]]
    coords_scaled = scaler.fit_transform(coords)
    
    n_init = 5
    progress_text = f"Đang xử lý..."
    my_bar = st.progress(0, text=progress_text)
    
    best_clf = None
    best_inertia = float('inf')

    try:
        for i in range(n_init):
            clf = KMeansConstrained(
                n_clusters=n_clusters, size_min=size_min, size_max=size_max,
                random_state=42 + i, n_init=1
            )
            clf.fit(coords_scaled)
            if clf.inertia_ < best_inertia:
                best_inertia = clf.inertia_
                best_clf = clf
            percent = int((i + 1) / n_init * 100)
            my_bar.progress((i + 1) / n_init, text=f"Đang xử lý...{percent}%")
        
        my_bar.empty()
        
        df_exploded['territory_id'] = best_clf.labels_ + 1
        final_labels = df_exploded.groupby('original_index')['territory_id'].agg(lambda x: x.mode()[0])
        df_run['territory_id'] = final_labels
        
        stats = df_run.groupby('territory_id').agg(
            count_kh=('territory_id', 'count'),
            sum_min=('workload_min', 'sum')
        ).reset_index()
        stats.columns = ['Tuyến (RouteID)', 'Số lượng KH', 'Workload_Total_Min']
        stats['Workload_Day'] = (stats['Workload_Total_Min'] / 60 / WORKING_DAYS).round(2)
        
        return df_run, stats
        
    except Exception as e:
        my_bar.empty()
        return None, str(e)

def generate_folium_map_tp(_df, _mapping, _time_matrix, mode="Chế độ 1"):
    if _df.empty: return None, None
    df_plot = _df.copy()
    lat_col, lon_col = _mapping['lat'], _mapping['lon']
    df_plot[lat_col] = pd.to_numeric(df_plot[lat_col], errors='coerce')
    df_plot[lon_col] = pd.to_numeric(df_plot[lon_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[lat_col, lon_col])
    
    if df_plot.empty: return None, None
    map_center = [df_plot[lat_col].mean(), df_plot[lon_col].mean()]
    
    m = folium.Map(location=map_center, zoom_start=11, prefer_canvas=True, tiles=ESRI_URL, attr=ESRI_ATTR)
    # ==============================================================
    
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080", "#FFA500"]
    unique_ids = sorted(df_plot['territory_id'].unique())
    color_map = {int(id): colors[(int(id) - 1) % len(colors)] for id in unique_ids}

    # LEGEND HTML (Vertical Box - Bottom Left)
    legend_html = ''' <div style="position: fixed; bottom: 30px; left: 30px; width: 120px; height: auto; 
                    border:2px solid grey; z-index:9999; font-size:12px; 
                    background-color:white; padding: 10px; opacity: 0.9;">
                    <b>Chú giải:</b><br>'''
    
    for rid in unique_ids:
        c = color_map.get(int(rid), 'gray')
        legend_html += f'<i style="background:{c}; width:10px; height:10px; display:inline-block; margin-right: 5px;"></i> Tuyến {rid}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    col_code = _mapping['customer_code']
    col_name = _mapping.get('customer_name')
    col_addr = _mapping.get('address')
    col_vol = _mapping.get('vol_ec')
    col_freq = _mapping.get('freq')
    col_type = _mapping.get('type')
    
    for _, row in df_plot.iterrows():
        # Enhanced Tooltip for TP
        tooltip_parts = [f"<b>KH: {row[col_code]}</b>", f"Tuyến: {row['territory_id']}"]
        if col_name and pd.notna(row.get(col_name)): tooltip_parts.append(f"Tên: {row[col_name]}")
        if col_addr and pd.notna(row.get(col_addr)): tooltip_parts.append(f"Đ/c: {row[col_addr]}")
        if col_vol and pd.notna(row.get(col_vol)): tooltip_parts.append(f"Vol: {row[col_vol]}")
        
        # Mode 1 also gets full tooltip now
        if col_freq and pd.notna(row.get(col_freq)): tooltip_parts.append(f"Tần suất: {row[col_freq]}")
        if col_type and pd.notna(row.get(col_type)): tooltip_parts.append(f"Phân loại: {row[col_type]}")
        
        if mode == "Chế độ 2":
            if col_type and pd.notna(row.get(col_type)):
                seg_val = str(row[col_type]).strip()
                key = seg_val if seg_val in _time_matrix else 'Mặc định/Trống'
                time_val = _time_matrix.get(key, 10.0)
                tooltip_parts.append(f"Thời gian: {time_val}p")

        tooltip_txt = "<br>".join(tooltip_parts)
        c = color_map.get(int(row['territory_id']), 'gray')

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4, color=c, fill=True, fill_color=c, fill_opacity=0.7,
            tooltip=tooltip_txt
        ).add_to(m)
    return m, map_center

# --- CODE 2 HELPER ---

REQUIRED_COLS_CUST = {
    'RouteID': ':red[RouteID (*)]', 
    'Customer code': ':red[Mã KH (*)]', 
    'Customer Name': 'Tên KH', 
    'Latitude': ':red[Latitude (Vĩ độ) (*)]', 
    'Longitude': ':red[Longitude (Kinh độ) (*)]', 
    'Frequency': ':red[Tần suất (*)]', 
    'Segment': ':red[Phân loại Segment (*)]'
}
REQUIRED_COLS_DIST = {
    'Distributor Code': ':red[Mã NPP (*)]', 
    'Distributor Name': 'Tên NPP', 
    'Latitude': ':red[Latitude (Vĩ độ) (*)]', 
    'Longitude': ':red[Longitude (Kinh độ) (*)]'
}

@st.cache_data
def calculate_haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_dynamic_travel_time(dist_km, speed_slow, speed_fast):
    speed_kmh = speed_slow if dist_km < 2.0 else speed_fast
    return (dist_km / speed_kmh) * 60

def calculate_dynamic_quantum(df_route, target_points=1000):
    total_time = (df_route['Visit Time (min)'] * df_route['Total_Visits_Count']).sum()
    raw_quantum = total_time / target_points
    return max(raw_quantum, 0.5)

def explode_data_by_quantum(df_week, quantum):
    df_process = df_week.copy()
    weighted_time = df_process['Visit Time (min)'] * df_process['Weight_Factor']
    df_process['quantum_points'] = np.ceil(weighted_time / quantum).fillna(1).astype(int)
    df_exploded = df_process.loc[df_process.index.repeat(df_process['quantum_points'])].copy()
    df_exploded['original_index'] = df_exploded.index
    df_exploded = df_exploded.reset_index(drop=True)
    return df_exploded, df_process['quantum_points'].sum()

def solve_saturday_strategy(df_exploded, total_points):
    coords = df_exploded[['Latitude', 'Longitude']].values.astype(np.float32)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    n_chunks = 11 if total_points >= 50 else 6
    avg_chunk_size = total_points / n_chunks
    min_size = max(1, int(avg_chunk_size * 0.90))
    max_size = int(avg_chunk_size * 1.10)
    if max_size * n_chunks < total_points: max_size = int(total_points / n_chunks) + 2
    try:
        kmeans = KMeansConstrained(n_clusters=n_chunks, size_min=min_size, size_max=max_size, random_state=42, n_init=10)
        chunk_labels = kmeans.fit_predict(coords_scaled)
    except:
        from sklearn.cluster import KMeans
        kmeans_fallback = KMeans(n_clusters=n_chunks, random_state=42, n_init=10)
        chunk_labels = kmeans_fallback.fit_predict(coords_scaled)
    df_exploded['Chunk_ID'] = chunk_labels
    chunk_centers = df_exploded.groupby('Chunk_ID')[['Latitude', 'Longitude']].mean()
    dists = np.sqrt((chunk_centers['Latitude'] - df_exploded['Latitude'].mean())**2 + (chunk_centers['Longitude'] - df_exploded['Longitude'].mean())**2)
    saturday_chunk_id = dists.idxmax()
    df_exploded['Day'] = np.where(df_exploded['Chunk_ID'] == saturday_chunk_id, 'Sat', None)
    weekday_mask = df_exploded['Chunk_ID'] != saturday_chunk_id
    if weekday_mask.any():
        weekday_coords = coords_scaled[weekday_mask]
        n_days = 5
        try:
            w_total = len(weekday_coords)
            w_avg = w_total / 5
            w_min = int(w_avg * 0.90)
            w_max = int(w_avg * 1.10)
            kmeans_5 = KMeansConstrained(n_clusters=n_days, size_min=w_min, size_max=w_max, random_state=42, n_init=10)
            day_labels_idx = kmeans_5.fit_predict(weekday_coords)
            days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
            df_exploded.loc[weekday_mask, 'Day'] = [days_map[i] for i in day_labels_idx]
        except:
             df_exploded.loc[weekday_mask, 'Day'] = 'Mon'
    return df_exploded

def collapse_to_original(df_exploded, original_df):
    final_assignments = df_exploded.groupby('original_index')['Day'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Mon')
    df_result = original_df.copy()
    df_result['Assigned_Day'] = final_assignments
    df_result['Assigned_Day'].fillna('Mon', inplace=True)
    return df_result

def build_time_matrix_haversine(locations, speed_slow, speed_fast):
    size = len(locations)
    matrix = np.zeros((size, size), dtype=int)
    lats = np.array([loc[0] for loc in locations])
    lons = np.array([loc[1] for loc in locations])
    for i in range(size):
        for j in range(size):
            if i == j: continue
            dist_km = calculate_haversine_distance_km(lats[i], lons[i], lats[j], lons[j])
            speed = speed_slow if dist_km < 2.0 else speed_fast
            matrix[i][j] = int((dist_km / speed) * 3600)
    return matrix.tolist()

def solve_tsp_final(visits, depot_coords, speed_slow, speed_fast, mode='closed', end_coords=None):
    if not visits: return []
    locations = [depot_coords] + [v['coords'] for v in visits]
    has_end_point = (mode == 'open' and end_coords is not None)
    if has_end_point: locations.append(end_coords) 
    num_locations = len(locations)
    time_matrix = build_time_matrix_haversine(locations, speed_slow, speed_fast)
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, [0], [num_locations - 1] if has_end_point else [0])
    routing = pywrapcp.RoutingModel(manager)
    def time_callback(from_index, to_index):
        return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 1 
    solution = routing.SolveWithParameters(search_parameters)
    ordered_visits = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                if has_end_point and node_index == num_locations - 1: pass 
                else: ordered_visits.append(visits[node_index - 1])
            index = solution.Value(routing.NextVar(index))
    return ordered_visits

def run_master_scheduler(df_cust, depot_coords, selected_route_ids, route_config_dict, visit_time_config, speed_config, progress_bar):
    SPEED_SLOW, SPEED_FAST = speed_config['slow'], speed_config['fast']
    df_cust = df_cust.copy()
    # === FIX: Ensure RouteID is string to match selection ===
    if 'RouteID' in df_cust.columns:
        df_cust['RouteID'] = df_cust['RouteID'].astype(str).str.strip()
        
    df_cust['Frequency'] = pd.to_numeric(df_cust['Frequency'], errors='coerce').fillna(1).round(0).astype(int)
    df_cust['Customer code'] = df_cust['Customer code'].astype(str).str.strip()
    
    # Ensure selection is also string
    selected_route_ids = [str(x) for x in selected_route_ids]
    
    df_cust_filtered = df_cust[df_cust['RouteID'].isin(selected_route_ids)].copy()
    
    if df_cust_filtered.empty:
        return pd.DataFrame() 

    cust_week_map = {} 
    f2_counter, f1_counter = 0, 0
    WEEKS = ['W1', 'W2', 'W3', 'W4']
    DAY_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    SPACING_MAP_F8 = {'Mon': 'Thu', 'Tue': 'Fri', 'Wed': 'Sat', 'Thu': 'Mon', 'Fri': 'Tue', 'Sat': 'Wed'}
    SPACING_MAP_F12 = {'Mon': ['Mon', 'Wed', 'Fri'], 'Tue': ['Tue', 'Thu', 'Sat'], 'Wed': ['Mon', 'Wed', 'Fri'], 'Thu': ['Tue', 'Thu', 'Sat'], 'Fri': ['Mon', 'Wed', 'Fri'], 'Sat': ['Tue', 'Thu', 'Sat']}
    for _, row in df_cust_filtered.iterrows():
        code, freq = row['Customer code'], row['Frequency']
        if freq == 2:
            cust_week_map[code] = ['W1', 'W3'] if f2_counter % 2 == 0 else ['W2', 'W4']
            f2_counter += 1
        elif freq == 1:
            cust_week_map[code] = [WEEKS[f1_counter % 4]]
            f1_counter += 1
    final_output_rows = []
    grouped = df_cust_filtered.groupby('RouteID')
    total_routes = len(selected_route_ids)
    for i, (route_id, route_df) in enumerate(grouped):
        progress_bar.progress(i / total_routes, text=f"Đang xử lý Route {route_id} - ({i+1}/{total_routes})")
        route_df['Visit Time (min)'] = route_df['Segment'].map(visit_time_config).fillna(visit_time_config.get('default', 10.0))
        route_df['Weight_Factor'] = 1.0
        for week in WEEKS:
            week_visits_all = [] 
            for _, row in route_df.iterrows():
                freq, code = row['Frequency'], row['Customer code']
                is_in_week = False
                num_visits = 0
                if freq >= 4: num_visits, is_in_week = int(freq // 4), True
                else:
                    if week in cust_week_map.get(code, []): is_in_week, num_visits = True, 1
                if is_in_week:
                    for v_i in range(int(num_visits)):
                        r = row.copy() 
                        r['Visit_ID_Internal'] = f"{code}_{week}_{v_i}" 
                        r['Visit_Order'] = v_i
                        r['Total_Visits_Count'] = num_visits
                        week_visits_all.append(r)
            if not week_visits_all: continue
            best_df, best_score = None, float('inf')
            for iteration in range(3):
                full_df = pd.DataFrame(week_visits_all)
                df_core = full_df[full_df['Visit_Order'] == 0].copy()
                quantum = calculate_dynamic_quantum(df_core, 1200)
                df_exploded, total_pts = explode_data_by_quantum(df_core, quantum)
                df_labeled = solve_saturday_strategy(df_exploded, total_pts)
                df_core_res = collapse_to_original(df_labeled, df_core)
                anchor_map = df_core_res.set_index('Customer code')['Assigned_Day'].to_dict()
                df_dependent = full_df[full_df['Visit_Order'] > 0].copy()
                if not df_dependent.empty:
                    dep_days = []
                    for _, r_d in df_dependent.iterrows():
                        anchor = anchor_map.get(r_d['Customer code'], 'Mon')
                        day = anchor
                        if r_d['Total_Visits_Count'] == 2 and r_d['Visit_Order'] == 1: day = SPACING_MAP_F8.get(anchor, 'Thu')
                        elif r_d['Total_Visits_Count'] == 3 and r_d['Visit_Order'] < 3: 
                            day = SPACING_MAP_F12.get(anchor, ['Mon', 'Wed', 'Fri'])[r_d['Visit_Order']]
                        dep_days.append(day)
                    df_dependent['Assigned_Day'] = dep_days
                df_combined = pd.concat([df_core_res, df_dependent])
                day_stats, total_work = {}, 0
                for day in DAY_ORDER:
                    d_visits = df_combined[df_combined['Assigned_Day'] == day]
                    if d_visits.empty: day_stats[day] = 0; continue
                    work = d_visits['Visit Time (min)'].sum()
                    day_stats[day] = work
                    total_work += work
                unit_work = total_work / 11
                max_dev = 0
                weights = {}
                for day, act in day_stats.items():
                    tgt = unit_work * (1 if day == 'Sat' else 2)
                    if tgt == 0: continue
                    ratio = act / tgt
                    max_dev = max(max_dev, abs(1 - ratio))
                    weights[day] = max(0.5, min(1 + (ratio - 1) * 0.7, 2.0))
                if max_dev < best_score: best_score, best_df = max_dev, df_combined.copy()
                if max_dev <= 1.10: break 
                for item in week_visits_all:
                    if item['Visit_Order'] == 0:
                        try:
                            day = df_core_res[df_core_res['Visit_ID_Internal'] == item['Visit_ID_Internal']]['Assigned_Day'].iloc[0]
                            item['Weight_Factor'] *= weights.get(day, 1.0)
                        except: pass
            for day in DAY_ORDER:
                d_visits = best_df[best_df['Assigned_Day'] == day]
                if d_visits.empty: continue
                tsp_in = []
                for _, row in d_visits.iterrows():
                    d = row.to_dict()
                    d['coords'] = (row['Latitude'], row['Longitude'])
                    tsp_in.append(d)
                end_cfg = route_config_dict.get(route_id)
                mode, end_c = ('open', end_cfg) if end_cfg else ('closed', None)
                ordered = solve_tsp_final(tsp_in, depot_coords, SPEED_SLOW, SPEED_FAST, mode, end_c)
                prev, seq, agg_time, agg_dist = depot_coords, 1, 0, 0
                for item in ordered:
                    curr = item['coords']
                    dist = calculate_haversine_distance_km(prev[0], prev[1], curr[0], curr[1])
                    travel = get_dynamic_travel_time(dist, SPEED_SLOW, SPEED_FAST)
                    agg_time += travel + item['Visit Time (min)']
                    agg_dist += dist
                    res = item.copy()
                    res.update({'RouteID': route_id, 'Week': week, 'Day': day, 'Week&Day': f"{week}-{day}",
                                'Sequence': seq, 'Travel Time (min)': round(travel, 2),
                                'Distance (km)': round(dist, 2), 'Total Workload (min)': round(agg_time, 2)})
                    for k in ['coords', 'angle', 'Weight_Factor', 'quantum_points']: 
                        if k in res: del res[k]
                    final_output_rows.append(res)
                    prev, seq = curr, seq+1
    if not final_output_rows: return pd.DataFrame()
    df_final = pd.DataFrame(final_output_rows)
    df_final['Day'] = pd.Categorical(df_final['Day'], categories=DAY_ORDER, ordered=True)
    return df_final.sort_values(by=['RouteID', 'Week', 'Day', 'Sequence'])

def recalculate_routes(df_edited, depot_coords, route_config, speed_config, impacted_groups=None):
    SPEED_SLOW, SPEED_FAST = speed_config['slow'], speed_config['fast']
    new_rows = []
    for (r_id, week, day), group in df_edited.groupby(['RouteID', 'Week', 'Day']):
        should_optimize = True
        if impacted_groups is not None:
            should_optimize = (r_id, week, day) in impacted_groups
        if should_optimize:
            tsp_input = []
            for _, row in group.iterrows():
                d = row.to_dict()
                d['coords'] = (row['Latitude'], row['Longitude'])
                tsp_input.append(d)
            end_cfg = route_config.get(r_id)
            mode, end_c = ('open', end_cfg) if end_cfg else ('closed', None)
            ordered = solve_tsp_final(tsp_input, depot_coords, SPEED_SLOW, SPEED_FAST, mode, end_c)
        else:
            ordered = [row.to_dict() for _, row in group.sort_values('Sequence').iterrows()]
            for item in ordered: item['coords'] = (item['Latitude'], item['Longitude'])
        prev, seq, agg_time, agg_dist = depot_coords, 1, 0, 0
        for item in ordered:
            curr = item['coords']
            dist = calculate_haversine_distance_km(prev[0], prev[1], curr[0], curr[1])
            travel = get_dynamic_travel_time(dist, SPEED_SLOW, SPEED_FAST)
            agg_time += travel + item['Visit Time (min)']
            agg_dist += dist
            res = item.copy()
            res.update({
                'Sequence': seq, 'Travel Time (min)': round(travel, 2),
                'Distance (km)': round(dist, 2), 'Total Workload (min)': round(agg_time, 2)
            })
            if 'coords' in res: del res['coords']
            new_rows.append(res)
            prev, seq = curr, seq+1
    return pd.DataFrame(new_rows)

def get_changed_visits(df_orig, df_curr):
    if df_orig is None or df_curr is None: return []
    df1 = df_orig.set_index('Visit_ID_Internal')[['Week', 'Day']].sort_index()
    df2 = df_curr.set_index('Visit_ID_Internal')[['Week', 'Day']].sort_index()
    common = df1.index.intersection(df2.index)
    diff = (df1.loc[common] != df2.loc[common]).any(axis=1)
    return diff[diff].index.tolist()

@st.cache_data
def create_template_excel(is_dist=False):
    if is_dist:
        data = { 'Mã NPP': ['12345678'], 'Tên NPP': ['NPP Thành Phát'], 'Vĩ độ (Latitude)': [10.7769], 'Kinh độ (Longitude)': [106.7009] }
    else:
        data = { 'RouteID': ['VN123456'], 'Mã KH': ['12345678'], 'Tên KH': ['Tạp Hóa A'], 'Vĩ độ (Latitude)': [10.77], 'Kinh độ (Longitude)': [106.70], 'Tần suất': [4], 'Phân loại Segment': ['Gold'], 'Thêm các cột khác tùy ý': [''] }
    df_sample = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df_sample.to_excel(writer, sheet_name='Template', index=False)
    return output.getvalue()

@st.cache_data
def create_template_v1():
    df = pd.DataFrame({ "Mã KH (*)": ["KH01", "KH02"], "Vĩ độ (Latitude) (*)": [10.7, 10.8], "Kinh độ (Longitude) (*)": [106.6, 106.7], "Tên KH": ["A", "B"], "Địa chỉ": ["HCM", "HCM"], "VolEC": [100, 200], "Tần suất": [4, 8], "Phân loại Segment": ["MT", "Cooler"] })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
    return output.getvalue()

@st.cache_data
def create_template_v2():
    df = pd.DataFrame({ "Mã KH (*)": ["KH01", "KH02"], "Vĩ độ (Latitude) (*)": [10.7, 10.8], "Kinh độ (Longitude) (*)": [106.6, 106.7], "Tên KH": ["A", "B"], "Địa chỉ": ["HCM", "HCM"], "VolEC": [100, 200], "Tần suất (*)": [4, 8], "Phân loại Segment (*)": ["MT", "Cooler"] })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
    return output.getvalue()

@st.cache_data
def to_excel_tp(df_export):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export_clean = df_export.copy()
        cols_to_drop = ['workload_min', 'weight_points', 'Trạng thái', 'Bỏ chọn']
        df_export_clean = df_export_clean.drop(columns=[c for c in cols_to_drop if c in df_export_clean.columns])
        df_export_clean.to_excel(writer, sheet_name='Details', index=False)
    return output.getvalue()

@st.cache_data
def to_excel_output(df_master):
    output = io.BytesIO()
    df_export = df_master.drop(columns=['Visit_ID_Internal'], errors='ignore').copy()
    
    # Remove technical columns if exist
    for c in ['Bỏ chọn', 'Đã sửa', 'Chọn', 'Trạng thái']:
        if c in df_export.columns:
            df_export = df_export.drop(columns=[c])
            
    df_export = df_export.sort_values(by=['RouteID', 'Week', 'Day', 'Sequence'])
    df_export['Agg_Dist'] = df_export.groupby(['RouteID', 'Week', 'Day'])['Distance (km)'].cumsum()
    df_export['Agg_Travel'] = df_export.groupby(['RouteID', 'Week', 'Day'])['Travel Time (min)'].cumsum()
    rename_map = { 'Day': 'Ngày', 'Week': 'Tuần', 'Week&Day': 'Ngày & Tuần', 'Sequence': 'Thứ tự', 'Distance (km)': 'Khoảng cách từ KH trước', 'Travel Time (min)': 'Thời gian di chuyển từ KH trước', 'Agg_Dist': 'Khoảng cách từ đầu ngày', 'Agg_Travel': 'Thời gian di chuyển từ đầu ngày', 'Visit Time (min)': 'Thời gian viếng thăm điểm bán', 'Total Workload (min)': 'Tổng thời gian làm việc từ đầu ngày' }
    df_export_final = df_export.rename(columns=rename_map)
    df_sum = df_master.groupby(['RouteID', 'Week', 'Day']).agg( Total_TIO_min=('Visit Time (min)', 'sum'), Total_TBO_min=('Travel Time (min)', 'sum'), Num_Customers=('Customer code', 'count') ).reset_index()
    df_sum['Total_Workload_min'] = df_sum['Total_TIO_min'] + df_sum['Total_TBO_min']
    df_sum['Total_TIO_h'] = (df_sum['Total_TIO_min'] / 60).round(2)
    df_sum['Total_TBO_h'] = (df_sum['Total_TBO_min'] / 60).round(2)
    df_sum['Total_Workload_h'] = (df_sum['Total_Workload_min'] / 60).round(2)
    sum_rename = { 'Week': 'Tuần', 'Day': 'Ngày', 'Total_TIO_h': 'Tổng thời gian viếng thăm (Giờ)', 'Total_TBO_h': 'Tổng thời gian di chuyển (Giờ)', 'Num_Customers': 'Số KH', 'Total_Workload_h': 'Tổng thời gian làm việc (Giờ)' }
    df_sum_final = df_sum.rename(columns=sum_rename)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export_final.to_excel(writer, sheet_name='Lịch viếng thăm', index=False)
        df_sum_final.to_excel(writer, sheet_name='Tổng quan', index=False)
    return output.getvalue()

@st.cache_data
def create_folium_map(df_filtered_dict, col_mapping):
    df_filtered = pd.DataFrame.from_dict(df_filtered_dict)
    if df_filtered.empty: return None
    center = [df_filtered['Latitude'].mean(), df_filtered['Longitude'].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles=ESRI_URL, attr=ESRI_ATTR)
    legend_html = ''' <div style="position: fixed; bottom: 30px; left: 30px; width: 80px; height: 130px; border:2px solid grey; z-index:9999; font-size:12px; background-color:white; padding: 10px; opacity: 0.9;"> <b>Chú giải:</b><br> <i style="background:red; width:10px; height:10px; display:inline-block;"></i> T2<br> <i style="background:green; width:10px; height:10px; display:inline-block;"></i> T3<br> <i style="background:blue; width:10px; height:10px; display:inline-block;"></i> T4<br> <i style="background:orange; width:10px; height:10px; display:inline-block;"></i> T5<br> <i style="background:purple; width:10px; height:10px; display:inline-block;"></i> T6<br> <i style="background:black; width:10px; height:10px; display:inline-block;"></i> T7<br> </div> '''
    m.get_root().html.add_child(folium.Element(legend_html))
    color_map = {'T2': 'red', 'T3': 'green', 'T4': 'blue', 'T5': 'orange', 'T6': 'purple', 'T7': 'black'}
    for (r, w, d), group in df_filtered.groupby(['RouteID', 'Week', 'Day']):
        color = color_map.get(d, 'gray')
        grp = group.sort_values('Sequence')
        folium.PolyLine(grp[['Latitude', 'Longitude']].values.tolist(), color=color, weight=3, opacity=0.7).add_to(m)
        for _, row in grp.iterrows():
            tooltip_parts = []
            for std_key, orig_label in col_mapping.items():
                if std_key in row and pd.notna(row[std_key]): tooltip_parts.append(f"<b>{orig_label}:</b> {row[std_key]}")
            # Ensure VP standard columns are shown too if not in col_mapping
            if 'Sequence' in row: tooltip_parts.append(f"<b>Thứ tự:</b> {row['Sequence']}")
            if 'Week' in row: tooltip_parts.append(f"<b>Tuần:</b> {row['Week']}")
            if 'Day' in row: tooltip_parts.append(f"<b>Ngày:</b> {row['Day']}")
            
            popup_txt = "<br>".join(tooltip_parts)
            icon_html = f"""<div style="background:{color};color:white;border-radius:50%;width:20px;height:20px;text-align:center;font-size:12px;font-weight:bold;line-height:20px;border:1px solid white;">{row['Sequence']}</div>"""
            folium.Marker( location=(row['Latitude'], row['Longitude']), icon=folium.DivIcon(html=icon_html), tooltip=popup_txt ).add_to(m)
    return m

@st.cache_data
def create_heatmap(df_dict, value_col, agg_mode, fmt="{:.1f}", title="Heatmap"):
    df_data = pd.DataFrame.from_dict(df_dict)
    weeks = ['W1', 'W2', 'W3', 'W4']
    days = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7']
    if agg_mode == 'count': pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc='count')
    elif agg_mode == 'sum_time': pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc=lambda x: x.sum()/60)
    elif agg_mode == 'mean_time': pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc=lambda x: x.mean()/60)
    elif agg_mode == 'mean_qty': pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc='mean')
    pivot = pivot.reindex(index=weeks, columns=days).fillna(0)
    pivot.index.name = None
    st.markdown(f"**{title}**")
    try:
        import matplotlib
        styled_df = pivot.style.format(fmt).background_gradient(cmap='RdYlGn_r', axis=None)
        st.dataframe(styled_df, height=140, use_container_width=True, column_config={col: st.column_config.Column(width="small") for col in days})
    except ImportError:
        st.dataframe(pivot.style.format(fmt), height=140, use_container_width=True, column_config={col: st.column_config.Column(width="small") for col in days})

def find_col_index(df_cols, target_name):
    for i, col in enumerate(df_cols):
        if str(col).strip().lower() == target_name.lower(): return i
    return 0

# ==========================================
# 3. SIDEBAR & WELCOME
# ==========================================

def render_sidebar():
    st.sidebar.markdown("### Thực hiện tác vụ:")
    
    check_tp = st.sidebar.checkbox("1. Chia địa bàn (Territory Planner)", value=st.session_state.global_state['config']['is_tp'])
    
    tp_mode_sel = "Chế độ 1"
    if check_tp:
        c_indent, c_content = st.sidebar.columns([0.15, 0.85])
        with c_content:
            st.write("Chia địa bàn theo:")
            mode_val = st.radio(
                "Mode", 
                ["Chế độ 1: Cân bằng Số lượng KH", "Chế độ 2: Cân bằng Workload"],
                label_visibility="collapsed"
            )
            tp_mode_sel = "Chế độ 1" if "Chế độ 1" in mode_val else "Chế độ 2"
        st.sidebar.write("")
        
    check_vp = st.sidebar.checkbox("2. Xếp lịch viếng thăm (Visit Planner)", value=st.session_state.global_state['config']['is_vp'])
    
    is_integrated = False
    if check_tp and check_vp:
        is_integrated = True
        st.sidebar.info("💡 Hệ thống sẽ dùng kết quả chia địa bàn để tự động xếp lịch viếng thăm.")
    
    st.sidebar.divider()
    
    if st.sidebar.button("Bắt đầu", type="primary"):
        if not check_tp and not check_vp:
            st.sidebar.error("Vui lòng chọn ít nhất 1 tác vụ!")
        else:
            # === HARD RESET LOGIC ===
            keys_to_reset = ['df', 'df_cust', 'df_dist', 'df_editing', 'df_final', 
                             'v1_df_edited', 'v2_df_edited', 'map_obj_Chế độ 1', 'map_obj_Chế độ 2']
            for k in keys_to_reset:
                if k in st.session_state: st.session_state[k] = None
            
            st.session_state.page = 'setup'
            st.session_state.stage = '1_upload'
            st.session_state.mapping_confirmed = False
            st.session_state.map_version += 1
            st.session_state.vp_msg = None
            st.session_state.vp_msg_type = None
            st.session_state.editor_filter_key = None
            st.session_state.show_download_options = False
            st.session_state.tp_confirm_clear = False
            st.session_state.vp_confirm_clear = False
            # ========================
            
            st.session_state.global_state['config']['is_tp'] = check_tp
            st.session_state.global_state['config']['is_vp'] = check_vp
            st.session_state.global_state['config']['is_integrated'] = is_integrated
            st.session_state.global_state['config']['tp_mode'] = tp_mode_sel
            st.session_state.global_state['has_started'] = True
            
            if is_integrated:
                st.session_state.global_state['step'] = 'input_integrated'
            elif check_tp:
                st.session_state.global_state['step'] = 'tp_setup'
            else:
                st.session_state.global_state['step'] = 'vp_input'
            
            st.rerun()

def render_welcome_screen():
    render_main_title()
    st.info("👈 Chọn tác vụ thực hiện ở thanh bên trái")

# ==========================================
# 4. MODULES UI (CODE GỐC 100% WORDING)
# ==========================================

def render_tp_ui(is_integrated=False):
    mode_key = st.session_state.global_state['config']['tp_mode']
    mode_key_slug = "chedo1" if mode_key == "Chế độ 1" else "chedo2"
    st.session_state.last_mode = mode_key
    
    # Get Current Integrated Step (if applicable)
    step = st.session_state.global_state['step']

    # --- PAGE 1: SETUP ---
    show_setup = False
    if not is_integrated and st.session_state.page == 'setup': show_setup = True
    if is_integrated and step == 'input_integrated': show_setup = True
    
    if show_setup:
        if not is_integrated:
            st.subheader("Bước 1: Tải dữ liệu")
            c_dl, c_up = st.columns([1, 3])
            with c_dl:
                template_data = create_template_v1() if mode_key == "Chế độ 1" else create_template_v2()
                st.download_button(f"📥 Tải template mẫu", template_data, f"Template_{mode_key}.xlsx", use_container_width=True)
            with c_up:
                uploaded_file = st.file_uploader("Upload", type=['xlsx', 'xls'], label_visibility="collapsed")

            if uploaded_file:
                try:
                    df = load_excel_file(uploaded_file)
                    st.session_state.df = df
                    all_cols = df.columns.tolist()
                    options_cols = ["[Bỏ qua]"] + all_cols
                    with st.form("column_mapping_form"):
                        c1, c2, c3 = st.columns(3)
                        def get_idx(name, lst, is_opt=False):
                            target = options_cols if is_opt else lst
                            idx = 0
                            if name in target: idx = target.index(name)
                            return idx

                        with c1:
                            st.caption("📍 Tọa độ & Mã KH")
                            cc_col = st.selectbox(":red[Code KH (*)]", all_cols, index=get_idx("Customer Code", all_cols))
                            lat_col = st.selectbox(":red[Lat (Vĩ độ) (*)]", all_cols, index=get_idx("Latitude", all_cols))
                            lon_col = st.selectbox(":red[Long (Kinh độ) (*)]", all_cols, index=get_idx("Longitude", all_cols))
                        with c2:
                            st.caption("📊 Phân loại")
                            if mode_key == "Chế độ 1":
                                freq_col = st.selectbox("Tần suất", options_cols, index=get_idx("Frequency", options_cols, True))
                                type_col = st.selectbox("Phân loại Segment", options_cols, index=get_idx("Segment", options_cols, True))
                            else:
                                freq_col = st.selectbox(":red[Tần suất (*)]", all_cols, index=get_idx("Frequency", all_cols))
                                type_col = st.selectbox(":red[Phân loại Segment (*)]", all_cols, index=get_idx("Segment", all_cols))
                        with c3:
                            st.caption("ℹ️ Thông tin phụ")
                            name_col = st.selectbox("Tên KH", options_cols, index=get_idx("Customer Name", options_cols, True))
                            addr_col = st.selectbox("Địa chỉ", options_cols, index=get_idx("Address", options_cols, True))
                            vol_col = st.selectbox("VolEC", options_cols, index=get_idx("VolEC", options_cols, True))
                        
                        st.write("") 
                        c_btn, c_msg = st.columns([1, 4]) 
                        with c_btn:
                            submitted = st.form_submit_button("✅ Xác nhận")
                        with c_msg:
                            if st.session_state.upload_msg:
                                if st.session_state.upload_msg_type == "success": 
                                    st.success(st.session_state.upload_msg, icon="✅")
                                else: 
                                    st.warning(st.session_state.upload_msg, icon="⚠️")

                    if submitted:
                        final_freq = None if (mode_key == "Chế độ 1" and freq_col == "[Bỏ qua]") else freq_col
                        final_type = None if (mode_key == "Chế độ 1" and type_col == "[Bỏ qua]") else type_col
                        mapping_local = { "customer_code": cc_col, "lat": lat_col, "lon": lon_col, "customer_name": None if name_col=="[Bỏ qua]" else name_col, "address": None if addr_col=="[Bỏ qua]" else addr_col, "vol_ec": None if vol_col=="[Bỏ qua]" else vol_col, "freq": final_freq, "type": final_type }
                        st.session_state.col_mapping = mapping_local
                        st.session_state.mapping_confirmed = True
                        if st.session_state.df is not None:
                            df_proc = st.session_state.df.copy()
                            total_rows = len(df_proc)
                            df_proc[lat_col] = pd.to_numeric(df_proc[lat_col], errors='coerce')
                            df_proc[lon_col] = pd.to_numeric(df_proc[lon_col], errors='coerce')
                            missing_coords_mask = df_proc[lat_col].isna() | df_proc[lon_col].isna()
                            n_missing_coords = missing_coords_mask.sum()
                            df_proc = df_proc.dropna(subset=[lat_col, lon_col])
                            n_dupes = df_proc.duplicated(subset=[cc_col]).sum()
                            df_proc = df_proc.drop_duplicates(subset=[cc_col], keep='first')
                            st.session_state.df = df_proc
                            
                            cleaned_count = len(df_proc)
                            details = []
                            if n_dupes > 0: details.append(f"Đã xóa {n_dupes} KH trùng lặp")
                            if n_missing_coords > 0: details.append(f"Đã xóa {n_missing_coords} KH trống tọa độ")
                            
                            if not details:
                                msg = f"Dữ liệu tải lên có {cleaned_count} KH (Không có KH trùng lặp hay trống tọa độ.)"
                            else:
                                msg = f"Dữ liệu tải lên có {cleaned_count} KH ({' và '.join(details)}.)"
                            
                            st.session_state.upload_msg = msg
                            
                            if n_dupes > 0 or n_missing_coords > 0: st.session_state.upload_msg_type = "warning"
                            else: st.session_state.upload_msg_type = "success"
                        st.rerun()
                except Exception as e: st.error(f"Lỗi file: {e}")
        else:
            if st.session_state.df is not None:
                # In integrated mode, validation msg already shown in previous block
                pass

        if st.session_state.get('mapping_confirmed') and st.session_state.df is not None:
            if is_integrated: st.subheader("Bước 2: Điều chỉnh Chia địa bàn")
            else: 
                st.divider()
                st.subheader("Bước 2: Điều chỉnh")
            
            if mode_key == "Chế độ 2":
                with st.expander("⏱️ Tùy chỉnh thời gian viếng thăm (Nhấn để mở)", expanded=True):
                    current_keys = list(st.session_state.time_matrix.keys())
                    cols = st.columns(len(current_keys))
                    for i, key in enumerate(current_keys):
                        with cols[i]:
                            val = st.session_state.time_matrix[key]
                            lbl = "Mặc định/trống (phút)" if key == 'Mặc định/Trống' else f"{key} (phút)"
                            new_val = st.number_input(lbl, min_value=1.0, value=float(val), step=1.0, format="%.0f", key=f"time_input_{key}")
                            st.session_state.time_matrix[key] = new_val

            c_route, c_min, c_max = st.columns(3)
            current_mapping = st.session_state.col_mapping
            with c_route: n_routes = st.number_input("Số tuyến (Routes)", 1, 100, 9)

            if mode_key == "Chế độ 1":
                avg_qty = len(st.session_state.df) // n_routes
                sug_min_v1 = int(avg_qty*0.9)
                sug_max_v1 = int(avg_qty*1.1)
                st.caption(f"Trung bình: ~{avg_qty} KH/tuyến")
                with c_min: min_v = st.number_input("Số KH tối thiểu/tuyến", 0, value=sug_min_v1)
                with c_max: max_v = st.number_input("Số KH tối đa/tuyến", 0, value=sug_max_v1)
                min_capacity_input, max_capacity_input = 0, 0
            else:
                temp_df = st.session_state.df.copy()
                time_matrix = st.session_state.time_matrix
                def quick_load(r):
                    try: f = float(r[current_mapping['freq']])
                    except: f = 1
                    c_type = str(r[current_mapping['type']]).strip()
                    key = c_type if c_type in time_matrix else 'Mặc định/Trống'
                    t = time_matrix.get(key, 10.0)
                    return f * t
                try:
                    total_min_all = temp_df.apply(quick_load, axis=1).sum()
                    avg_hours_day = (total_min_all / 60 / WORKING_DAYS) / n_routes
                    st.caption(f"Trung bình: ~{avg_hours_day:.1f} giờ/tuyến/ngày")
                    with c_min:
                        min_day = st.number_input("Số giờ tối thiểu/tuyến/ngày", min_value=0.5, max_value=24.0, value=round(avg_hours_day*0.9, 1), step=0.5)
                    with c_max:
                        max_day = st.number_input("Số giờ tối đa/tuyến/ngày", min_value=0.5, max_value=24.0, value=round(avg_hours_day*1.1, 1), step=0.5)
                    min_capacity_input = min_day * 60 * WORKING_DAYS
                    max_capacity_input = max_day * 60 * WORKING_DAYS
                    min_v, max_v = 0, 0
                except Exception as e:
                    st.error(f"Lỗi tính toán workload: {e}")
                    min_capacity_input, max_capacity_input = 0, 0

            st.write("")
            if st.button("🚀 Bắt đầu phân tuyến", type="primary", use_container_width=True):
                df_input = st.session_state.df
                res_df, res_stats = None, None
                err = None
                final_mapping = st.session_state.col_mapping
                
                if mode_key == "Chế độ 1":
                    res_df, res_stats = run_territory_planning_v1(df_input, final_mapping['lat'], final_mapping['lon'], n_routes, min_v, max_v)
                    if res_df is None: err = res_stats
                else:
                    if not final_mapping.get('freq') or not final_mapping.get('type'): err = "Lỗi: Chế độ 2 cần cột Tần suất & Phân loại."
                    else:
                        res_df, res_stats = run_territory_planning_v2(df_input, final_mapping['lat'], final_mapping['lon'], final_mapping['freq'], final_mapping['type'], st.session_state.time_matrix, n_routes, min_capacity_input, max_capacity_input)
                        if res_df is None: err = res_stats

                if err: st.error(err)
                elif res_df is not None:
                    st.session_state.map_version = 0 
                    st.session_state.col_widths = {}
                    st.session_state.v1_map_snapshot = None
                    st.session_state.v2_map_snapshot = None
                    if mode_key == "Chế độ 1":
                        st.session_state.v1_df_edited = res_df.copy()
                        st.session_state.v1_report = res_stats.copy()
                        st.session_state.v1_df_original = res_df.copy()
                    else:
                        st.session_state.v2_df_edited = res_df.copy()
                        st.session_state.v2_report = res_stats.copy()
                        st.session_state.v2_df_original = res_df.copy()
                    
                    if is_integrated:
                        st.session_state.global_state['step'] = 'tp_result_integrated'
                    else:
                        st.session_state.page = 'result'
                    
                    st.rerun()

    # --- PAGE 2: RESULT (Shared for Integrated & Standalone) ---
    show_result = False
    if not is_integrated and st.session_state.page == 'result': show_result = True
    if is_integrated and step == 'tp_result_integrated': show_result = True

    if show_result:
        mode_key = st.session_state.last_mode
        st.subheader(f"Kết quả Chia địa bàn ({mode_key})")
        mapping = st.session_state.col_mapping
        
        if mode_key == "Chế độ 1":
            key_original, key_saved = 'v1_df_original', 'v1_df_edited'
            key_snapshot = 'v1_map_snapshot'
            current_df_edited = st.session_state.v1_df_edited
            df_original = st.session_state.v1_df_original
        else:
            key_original, key_saved = 'v2_df_original', 'v2_df_edited'
            key_snapshot = 'v2_map_snapshot'
            current_df_edited = st.session_state.v2_df_edited
            df_original = st.session_state.v2_df_original

        df_saved = st.session_state[key_saved].copy()

        if st.session_state.get(key_snapshot) is None or st.session_state.map_needs_refresh:
            m_snapshot, _ = generate_folium_map_tp(df_saved, mapping, st.session_state.time_matrix, mode=mode_key)
            st.session_state[key_snapshot] = m_snapshot
            st.session_state.map_needs_refresh = False

        # Scorecard
        display_data = []
        if mode_key == "Chế độ 1":
            curr_stats = df_saved['territory_id'].value_counts().sort_index()
            display_data = [{'id': r, 'val': int(curr_stats.get(r, 0)), 'count': int(curr_stats.get(r, 0))} for r in sorted(df_saved['territory_id'].unique())]
        else:
            grouped_v2 = df_saved.groupby('territory_id').agg(count=('territory_id', 'count'), workload_min=('workload_min', 'sum')).sort_index()
            for route_id in sorted(df_saved['territory_id'].unique()):
                if route_id in grouped_v2.index:
                    row_data = grouped_v2.loc[route_id]
                    val_h_day = round(row_data['workload_min'] / 60 / WORKING_DAYS, 1)
                    display_data.append({'id': route_id, 'val': val_h_day, 'count': int(row_data['count'])})
                else: display_data.append({'id': route_id, 'val': 0, 'count': 0})

        html_items = []
        for item in display_data:
            route_id = item['id']
            val = item['val']
            current_kh = item['count']
            orig_kh = df_original[df_original['territory_id'] == route_id].shape[0]
            arrow = " 🔼" if current_kh > orig_kh else (" 🔽" if current_kh < orig_kh else "")
            if mode_key == "Chế độ 1": content = f"<b>Tuyến {route_id}{arrow}</b><br>{val} KH"
            else: content = f"<b>Tuyến {route_id}{arrow}</b><br>{item['count']} KH<br>{val} h/ngày"
            html_items.append(f'<span style="display: inline-block; padding: 5px 15px; text-align: center; border-right: 1px solid #eee; line-height: 1.4;">{content}</span>')
        st.markdown(f"""<div style="overflow-x: auto; white-space: nowrap; display: inline-block; max-width: 100%; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9;">{''.join(html_items)}</div>""", unsafe_allow_html=True)
        st.divider()

        # UI Layout Map & Edit - UPDATED TO MATCH VP
        if 'editor_filter_mode' not in st.session_state: st.session_state.editor_filter_mode = 'all'
        if 'editor_filter_key' not in st.session_state: st.session_state.editor_filter_key = None
        if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False
        map_key_obj = f"map_obj_{mode_key}"
        
        c_map, c_edit = st.columns([3, 2])
        
        with c_map:
            # STATIC MAP LOGIC MOVED INSIDE COLUMN
            map_data = None
            if st.session_state[key_snapshot]:
                 map_data = st_folium(st.session_state[key_snapshot], center=[df_saved[mapping['lat']].mean(), df_saved[mapping['lon']].mean()], zoom=11, height=550, returned_objects=["last_object_clicked"], key=f"map_v{st.session_state.map_version}", use_container_width=True)

            # Logic for click processing
            if map_data and map_data.get("last_object_clicked"):
                clicked_obj = map_data["last_object_clicked"]
                c_lat, c_lng = clicked_obj['lat'], clicked_obj['lng']
                mask = (np.isclose(df_saved[mapping['lat']], c_lat, atol=1e-5)) & (np.isclose(df_saved[mapping['lon']], c_lng, atol=1e-5))
                found_rows = df_saved[mask]
                if not found_rows.empty:
                    found_code = found_rows.iloc[0][mapping['customer_code']]
                    if st.session_state.editor_filter_key != found_code:
                        st.session_state.editor_filter_mode = 'single'
                        st.session_state.editor_filter_key = found_code
                        st.rerun()

        with c_edit:
            st.subheader("🛠️ Chỉnh sửa thủ công")
            
            df_display_source = df_saved.copy()
            orig_map_dict = dict(zip(df_original[mapping['customer_code']], df_original['territory_id']))
            
            # Vectorization
            original_routes = df_display_source[mapping['customer_code']].map(orig_map_dict)
            current_routes = df_display_source['territory_id'].astype(int)
            original_routes = original_routes.fillna(current_routes).astype(int)
            df_display_source['Trạng thái'] = np.where(current_routes != original_routes, "✏️", "")
            
            filter_mode = st.session_state.editor_filter_mode
            if filter_mode == 'single' and st.session_state.editor_filter_key:
                df_display = df_display_source[df_display_source[mapping['customer_code']] == st.session_state.editor_filter_key]
                st.info(f"KH: {st.session_state.editor_filter_key}")
            elif filter_mode == 'changed':
                df_display = df_display_source[df_display_source['Trạng thái'] != ""]
                st.warning(f"Đang lọc {len(df_display)} KH thay đổi.")
            else: df_display = df_display_source

            cols_cfg = {}
            # Fix error for None columns
            show_cols = ['Trạng thái', mapping['customer_code']]
            if mapping.get('customer_name'): show_cols.append(mapping['customer_name'])
            show_cols.append('territory_id')

            if mode_key == "Chế độ 2":
                if mapping.get('freq'): show_cols.append(mapping['freq'])
                if mapping.get('type'): show_cols.append(mapping['type'])
            
            cols_cfg['Trạng thái'] = st.column_config.TextColumn("Trạng thái", width=80, disabled=True)
            cols_cfg[mapping['customer_code']] = st.column_config.TextColumn("Mã KH", width=100, disabled=True)
            cols_cfg['territory_id'] = st.column_config.SelectboxColumn("Tuyến", options=[int(x) for x in sorted(df_saved['territory_id'].unique())], required=True, width=80)
            if mapping.get('customer_name'): cols_cfg[mapping['customer_name']] = st.column_config.TextColumn("Tên KH", width=180, disabled=True)
            if mode_key == "Chế độ 2":
                if mapping.get('freq'): cols_cfg[mapping['freq']] = st.column_config.TextColumn("Tần suất", width=70, disabled=True)
                if mapping.get('type'): cols_cfg[mapping['type']] = st.column_config.TextColumn("Phân loại Segment", width=100, disabled=True)
            
            editor_key = f"editor_{mode_key}_{filter_mode}_{st.session_state.editor_filter_key}"
            
            edited_data_sub = st.data_editor(df_display, column_config=cols_cfg, column_order=show_cols, use_container_width=False, hide_index=True, height=400, key=editor_key)

            # Control Logic
            has_unsaved_changes = False
            if not edited_data_sub['territory_id'].equals(df_display['territory_id']):
                has_unsaved_changes = True
            
            # --- NEW LAYOUT: 1 Row [1, 1.2, 0.8] ---
            c_update, c_filter_change, c_clear = st.columns([1, 1.2, 0.8])
            
            with c_update: 
                if st.button("💾 Cập nhật", use_container_width=True, type="primary"):
                    new_map = dict(zip(edited_data_sub[mapping['customer_code']], edited_data_sub['territory_id']))
                    def update_route_logic(row):
                        code = row[mapping['customer_code']]
                        return new_map.get(code, row['territory_id'])
                    df_to_save = df_saved.copy()
                    df_to_save['territory_id'] = df_to_save.apply(update_route_logic, axis=1)
                    st.session_state[key_saved] = df_to_save
                    st.session_state.map_needs_refresh = True # Refresh map on Save
                    st.session_state.map_version += 1
                    st.session_state.tp_confirm_clear = False
                    st.success("Đã cập nhật!")
                    st.rerun()

            with c_filter_change: 
                is_single_filter = (st.session_state.editor_filter_mode == 'single')
                has_global_changes = not df_saved['territory_id'].equals(df_original['territory_id'])
                
                if st.button("🌪️ Lọc KH đã sửa", use_container_width=True, disabled=(filter_mode == 'changed' or is_single_filter or not has_global_changes)):
                    st.session_state.editor_filter_mode = 'changed'
                    st.session_state.editor_filter_key = None
                    st.rerun()
                    
            with c_clear: 
                if st.button("✖ Bỏ lọc", use_container_width=True, disabled=(filter_mode == 'all')):
                    if has_unsaved_changes:
                        st.session_state.tp_confirm_clear = True
                        st.rerun()
                    else:
                        st.session_state.editor_filter_mode = 'all'
                        st.session_state.editor_filter_key = None
                        st.session_state.map_version += 1 # Force map reset when clearing filter
                        st.rerun()

            # --- WARNING AREA (Unified with VP) ---
            if st.session_state.tp_confirm_clear:
                st.caption("⚠️ Bạn có thay đổi chưa lưu.")
                c_save_clear, c_discard_clear = st.columns(2)
                
                # Button: Save & Clear
                if c_save_clear.button("Lưu & Bỏ lọc", type="secondary", use_container_width=True):
                     new_map = dict(zip(edited_data_sub[mapping['customer_code']], edited_data_sub['territory_id']))
                     def update_route_logic(row):
                         code = row[mapping['customer_code']]
                         return new_map.get(code, row['territory_id'])
                     df_to_save = df_saved.copy()
                     df_to_save['territory_id'] = df_to_save.apply(update_route_logic, axis=1)
                     st.session_state[key_saved] = df_to_save
                     st.session_state.map_needs_refresh = True
                     
                     st.session_state.editor_filter_mode = 'all'
                     st.session_state.editor_filter_key = None
                     st.session_state.tp_confirm_clear = False
                     st.session_state.map_version += 1
                     st.rerun()
                
                # Button: Discard & Clear
                if c_discard_clear.button("Không lưu & Bỏ lọc", type="secondary", use_container_width=True):
                    st.session_state.editor_filter_mode = 'all'
                    st.session_state.editor_filter_key = None
                    st.session_state.tp_confirm_clear = False
                    st.session_state.map_version += 1 # Force Map Reset
                    st.rerun()

            st.divider()
            
            if not st.session_state.confirm_reset:
                if st.button("🔄 Hủy bỏ & Reset", type="secondary", use_container_width=True):
                    st.session_state.confirm_reset = True
                    st.rerun()
            else:
                st.error("Quay về phiên bản trước khi chỉnh sửa?")
                c_yes, c_no = st.columns(2)
                if c_yes.button("✅ Đồng ý", use_container_width=True, type="primary"):
                    st.session_state[key_saved] = df_original.copy()
                    st.session_state.editor_filter_mode = 'all'
                    st.session_state.editor_filter_key = None
                    st.session_state.confirm_reset = False
                    st.session_state.just_reset = True 
                    st.session_state.map_needs_refresh = True
                    st.session_state.map_version += 1
                    st.success("Đã Reset!")
                    st.rerun()
                if c_no.button("❌ Hủy", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()

        # EXPORT & NAVIGATION
        st.divider()
        c_back, c_empty, c_download, c_next = st.columns([1, 2, 1, 1.5])
        
        with c_back: st.button("⬅️ Quay lại", on_click=lambda: st.session_state.update(page='setup'))

        with c_download:
            # TP Export Logic (Standalone)
            if not is_integrated:
                buffer = to_excel_tp(df_saved)
                st.download_button(f"📥 Tải file excel kết quả", buffer, f"Result_Chiadiaban_{mode_key_slug}.xlsx", use_container_width=True)

        # INTEGRATION BRIDGE
        if is_integrated:
            with c_next:
                if st.button("Tiếp tục Bước 3: Xếp lịch viếng thăm", type="primary"):
                    st.session_state.df_cust = df_saved.copy()
                    if 'RouteID' in st.session_state.df_cust.columns:
                        st.session_state.df_cust.drop(columns=['RouteID'], inplace=True)
                        
                    st.session_state.df_cust = st.session_state.df_cust.rename(columns={mapping['customer_code']: 'Customer code', 'territory_id': 'RouteID'})
                    st.session_state.df_cust['RouteID'] = st.session_state.df_cust['RouteID'].astype(str)
                    
                    st.session_state.global_state['step'] = 'vp_process'
                    st.rerun()

def render_vp_ui(is_integrated=False):
    step = st.session_state.global_state['step']
    
    # --- SCREEN 1: DATA INPUT ---
    if step == 'vp_input' or step == 'input_integrated':
        st.subheader("Bước 1: Tải dữ liệu")
        c1, c2 = st.columns(2)
        c1.download_button("📥 Tải Template Customers", create_template_excel(is_dist=False), "Customers_Template.xlsx")
        c2.download_button("📥 Tải Template Distributors", create_template_excel(is_dist=True), "Distributors_Template.xlsx")
        
        u1, u2 = st.columns(2)
        up_cust = u1.file_uploader("Upload File Customers", type=['xlsx'])
        up_dist = u2.file_uploader("Upload File Distributors", type=['xlsx'])
        
        if up_cust and up_dist:
            df_c, df_d = pd.read_excel(up_cust), pd.read_excel(up_dist)
            st.markdown("---")
            with st.form("mapping"):
                c1, c2 = st.columns(2)
                
                # --- HIDE RouteID for Integrated Mode ---
                cols_cust_active = REQUIRED_COLS_CUST.copy()
                if is_integrated and 'RouteID' in cols_cust_active:
                    del cols_cust_active['RouteID']
                
                map_c = {k: c1.selectbox(f"File Customers: {cols_cust_active[k]}", df_c.columns, index=find_col_index(df_c.columns, k)) for k in cols_cust_active}
                map_d = {k: c2.selectbox(f"File Distributors: {REQUIRED_COLS_DIST[k]}", df_d.columns, index=find_col_index(df_d.columns, k)) for k in REQUIRED_COLS_DIST}
                
                c_btn, c_msg = st.columns([1, 3])
                
                with c_btn:
                    submitted = st.form_submit_button("✅ Xác nhận & Kiểm tra")
                
                with c_msg:
                    if st.session_state.vp_msg:
                        if st.session_state.vp_msg_type == 'success':
                            st.success(st.session_state.vp_msg, icon="✅")
                        else:
                            st.warning(st.session_state.vp_msg, icon="⚠️")

                if submitted:
                    if len(set(map_c.values())) < len(map_c):
                        st.error("⚠️ Lỗi (File Customers): Bạn đang chọn 1 cột cho nhiều trường dữ liệu khác nhau. Vui lòng kiểm tra lại!")
                        st.stop()
                    if len(set(map_d.values())) < len(map_d):
                        st.error("⚠️ Lỗi (File Distributors): Bạn đang chọn 1 cột Excel cho nhiều trường dữ liệu khác nhau. Vui lòng kiểm tra lại!")
                        st.stop()

                    st.session_state.col_map_main = map_c 
                    df_c = df_c.rename(columns={v: k for k, v in map_c.items()})
                    df_d = df_d.rename(columns={v: k for k, v in map_d.items()})
                    
                    if 'Customer code' in df_c.columns: df_c['Customer code'] = df_c['Customer code'].astype(str).str.strip()
                    
                    # Only process RouteID if not integrated
                    if not is_integrated:
                        if 'RouteID' in df_c.columns: df_c['RouteID'] = df_c['RouteID'].astype(str).str.strip()
                    
                    if 'Distributor Code' in df_d.columns: df_d['Distributor Code'] = df_d['Distributor Code'].astype(str).str.strip()
                    
                    # VALIDATION LOGIC
                    total_rows = len(df_c)
                    missing_coords_mask = df_c['Latitude'].isna() | df_c['Longitude'].isna()
                    n_missing_coords = missing_coords_mask.sum()
                    df_c = df_c.dropna(subset=['Latitude', 'Longitude'])
                    n_dupes = df_c.duplicated(subset=['Customer code']).sum()
                    if n_dupes > 0: st.warning("Loại bỏ KH trùng lặp.")
                    df_c = df_c.drop_duplicates('Customer code', keep='first')
                    
                    msg = f"Dữ liệu tải lên có {total_rows} KH. Có {n_dupes} KH trùng lặp, {n_missing_coords} KH trống tọa độ."
                    st.session_state.vp_msg = msg
                    st.session_state.vp_msg_type = 'warning' if (n_dupes > 0 or n_missing_coords > 0) else 'success'
                    
                    st.session_state.df_cust = df_c
                    st.session_state.df_dist = df_d
                    
                    # BRIDGE LOGIC
                    if is_integrated:
                        # Map columns for Code 1
                        st.session_state.df = df_c.copy()
                        st.session_state.col_mapping = {
                            "customer_code": 'Customer code', "lat": 'Latitude', "lon": 'Longitude',
                            "customer_name": 'Customer Name', "address": None, "vol_ec": None,
                            "freq": 'Frequency', "type": 'Segment'
                        }
                        st.session_state.mapping_confirmed = True
                        st.rerun()
                    else:
                        st.session_state.global_state['step'] = 'vp_process'
                        st.rerun()

        # --- INTEGRATED: Show TP Setup Below ---
        if is_integrated and st.session_state.get('mapping_confirmed'):
            st.divider()
            render_tp_ui(is_integrated=True) # Re-use TP UI render logic but in Integrated Context

    # --- SCREEN 2: CONFIGURATION ---
    elif step == 'vp_process':
        if is_integrated: st.subheader("Bước 3: Điều chỉnh Xếp lịch viếng thăm")
        else: st.subheader("Bước 2: Điều chỉnh")
        
        unique_dist = st.session_state.df_dist.drop_duplicates(subset=['Distributor Code'])
        dist_opts = unique_dist.apply(lambda x: f"{x['Distributor Code']} - {x['Distributor Name']}", axis=1)
        sel_dist = st.selectbox("Chọn Nhà Phân Phối:", dist_opts)
        
        sel_code = sel_dist.split(' - ')[0]
        depot_row = st.session_state.df_dist[st.session_state.df_dist['Distributor Code'] == sel_code].iloc[0]
        st.session_state.depot_coords = (depot_row['Latitude'], depot_row['Longitude'])
        
        all_routes = sorted(st.session_state.df_cust['RouteID'].unique().astype(str))
        sel_routes = st.multiselect("Chọn RouteID:", all_routes, default=all_routes[:1])
        
        route_end_point_configs = {}
        if sel_routes:
            st.markdown("**Chọn Điểm Kết Thúc ngày làm việc:**")
            for r_id in sel_routes:
                c1, c2, c3 = st.columns([1, 2, 3])
                c1.write(f"🏷️ **Tuyến {r_id}**")
                mode = c2.selectbox(f"Chế độ {r_id}", ["Quay về NPP", "Kết thúc tại 1 KH"], label_visibility="collapsed")
                if "Kết thúc" in mode:
                    custs = st.session_state.df_cust[st.session_state.df_cust['RouteID'].astype(str) == str(r_id)]
                    opts = custs.apply(lambda x: f"{x['Customer code']} - {x.get('Customer Name','')}", axis=1)
                    sel_c = c3.selectbox(f"Chọn KH {r_id}", opts, label_visibility="collapsed")
                    if sel_c:
                        c_row = custs[custs['Customer code'] == sel_c.split(' - ')[0]].iloc[0]
                        route_end_point_configs[r_id] = (c_row['Latitude'], c_row['Longitude'])
                else:
                    route_end_point_configs[r_id] = None

        with st.expander("⚙️ Tùy chỉnh Tốc độ & Thời gian (Nhấn để mở)", expanded=False):
            c1, c2 = st.columns(2)
            s_slow = c1.number_input("KH cách nhau dưới 2km (đơn vị: km/h)", min_value=10, max_value=60, value=20, step=5)
            s_fast = c2.number_input("KH cách nhau trên 2km (đơn vị: km/h)", min_value=30, max_value=100, value=40, step=5)
            st.write("Thời gian viếng thăm (phút):")
            cols = st.columns(6)
            vt_cfg = {}
            for i, (k, v) in enumerate({'MT':19.5, 'Cooler':18.0, 'Gold':9.0, 'Silver':7.8, 'Bronze':6.8, 'default':10.0}.items()):
                lbl = k
                if k == 'default': lbl = "Mặc định/trống (phút)"
                vt_cfg[k] = cols[i].number_input(lbl, 0.0, 60.0, v, step=1.0)
        
        c_back, c_run = st.columns([1, 5])
        if c_back.button("⬅️ Quay lại"):
            if is_integrated: st.session_state.global_state['step'] = 'tp_result_integrated'
            else: st.session_state.global_state['step'] = 'vp_input'
            st.rerun()
        
        if c_run.button("🚀 Chạy Xếp lịch viếng thăm", type="primary", disabled=not sel_routes):
            pb = st.progress(0, "Đang xử lý...")
            try:
                st.session_state.route_cfg = route_end_point_configs
                st.session_state.speed_cfg = {'slow': s_slow, 'fast': s_fast}
                
                df_res = run_master_scheduler(
                    st.session_state.df_cust, st.session_state.depot_coords, sel_routes,
                    route_end_point_configs, vt_cfg, st.session_state.speed_cfg, pb
                )
                
                if df_res.empty:
                    st.warning("Không tìm thấy lịch trình phù hợp. Vui lòng kiểm tra lại cấu hình hoặc dữ liệu (Tần suất, Segment...).")
                else:
                    day_map = {'Mon': 'T2', 'Tue': 'T3', 'Wed': 'T4', 'Thu': 'T5', 'Fri': 'T6', 'Sat': 'T7'}
                    df_res['Day'] = df_res['Day'].map(day_map)
                    
                    st.session_state.df_final = df_res.copy()
                    st.session_state.df_editing = df_res.copy() 
                    st.session_state.global_state['step'] = 'vp_result'
                    st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {e}")
                import traceback
                st.text(traceback.format_exc())

    # --- SCREEN 3: DASHBOARD & EDITOR ---
    elif step == 'vp_result':
        st.subheader("Kết quả")
        
        all_r = ['All Routes'] + sorted(st.session_state.df_editing['RouteID'].unique().tolist())
        c_sel, _ = st.columns([2, 8])
        with c_sel: sel_r_view = st.selectbox("Xem Route:", all_r)

        df_view = st.session_state.df_editing.copy()
        df_view['Workload_Single_Min'] = df_view['Visit Time (min)'] + df_view['Travel Time (min)']
        
        heatmap_data = {}
        agg_mode_cust, agg_mode_time = 'count', 'sum_time'
        col_cust, col_work, col_visit, col_travel = 'Customer code', 'Workload_Single_Min', 'Visit Time (min)', 'Travel Time (min)'
        
        if sel_r_view == 'All Routes':
            df_route_daily = df_view.groupby(['RouteID', 'Week', 'Day']).agg(
                num_cust=('Customer code', 'count'),
                total_work=('Workload_Single_Min', 'sum'),
                total_visit=('Visit Time (min)', 'sum'),
                total_travel=('Travel Time (min)', 'sum')
            ).reset_index()
            heatmap_data = df_route_daily.to_dict('list')
            agg_mode_cust, agg_mode_time = 'mean_qty', 'mean_time'
            col_cust, col_work, col_visit, col_travel = 'num_cust', 'total_work', 'total_visit', 'total_travel'
        else:
            df_view = df_view[df_view['RouteID'] == sel_r_view]
            heatmap_data = df_view.to_dict('list')
            
        df_editor_view = df_view.copy()
        
        r1c1, r1c2 = st.columns(2)
        with r1c1: create_heatmap(heatmap_data, col_cust, agg_mode_cust, "{:.0f}", "Số lượng KH/ngày (TB)" if sel_r_view == 'All Routes' else "Số lượng KH/ngày")
        with r1c2: create_heatmap(heatmap_data, col_work, agg_mode_time, "{:.1f}", "Tổng giờ làm việc/ngày (TB)" if sel_r_view == 'All Routes' else "Tổng giờ làm việc/ngày")
        
        r2c1, r2c2 = st.columns(2)
        with r2c1: create_heatmap(heatmap_data, col_visit, agg_mode_time, "{:.1f}", "Tổng giờ viếng thăm/ngày (TB)" if sel_r_view == 'All Routes' else "Tổng giờ viếng thăm/ngày")
        with r2c2: create_heatmap(heatmap_data, col_travel, agg_mode_time, "{:.1f}", "Tổng giờ di chuyển/ngày (TB)" if sel_r_view == 'All Routes' else "Tổng giờ di chuyển/ngày")
        
        st.markdown("---")

        col_map, col_edit = st.columns([3, 2])
        
        with col_map:
            mf1, mf2 = st.columns([1, 2])
            all_weeks = sorted(df_view['Week'].unique())
            def_week = ['W1'] if 'W1' in all_weeks else all_weeks
            weeks = mf1.multiselect("Lọc Tuần:", all_weeks, default=def_week)
            days = mf2.multiselect("Lọc Ngày:", ['T2','T3','T4','T5','T6','T7'], default=['T2','T3','T4','T5','T6','T7'])
            
            df_map = df_view[(df_view['Week'].isin(weeks)) & (df_view['Day'].isin(days))]
            st.caption("💡 Click vào điểm trên bản đồ để sửa nhanh bên phải.")
            
            # STATIC MAP FOR VP
            map_data = st_folium(
                create_folium_map(df_map.to_dict('list'), st.session_state.col_map_main), 
                height=550, use_container_width=True,
                key=f"folium_map_{st.session_state.map_version}",
                returned_objects=["last_object_clicked"]
            )
            
            if map_data and map_data.get("last_object_clicked"):
                lat, lng = map_data["last_object_clicked"]['lat'], map_data["last_object_clicked"]['lng']
                dist_sq = (st.session_state.df_editing['Latitude'] - lat)**2 + (st.session_state.df_editing['Longitude'] - lng)**2
                min_idx = dist_sq.idxmin()
                min_val = dist_sq.min()
                if min_val < 1e-6:
                    clicked_code = st.session_state.df_editing.loc[min_idx, 'Customer code']
                    if st.session_state.map_clicked_code != clicked_code:
                        st.session_state.map_clicked_code = clicked_code
                        st.session_state.editor_filter_mode = 'single'
                        st.rerun()

        with col_edit:
            st.subheader("🛠️ Chỉnh sửa Thủ công")
            
            changed_ids = get_changed_visits(st.session_state.df_final, st.session_state.df_editing)
            # Vectorization
            df_editor_view['Trạng thái'] = np.where(df_editor_view['Visit_ID_Internal'].isin(changed_ids), "✏️", "")

            if st.session_state.editor_filter_mode == 'single' and st.session_state.map_clicked_code:
                df_editor_view = df_editor_view[df_editor_view['Customer code'] == st.session_state.map_clicked_code]
                st.info(f"Đang sửa KH: {st.session_state.map_clicked_code}")
            elif st.session_state.editor_filter_mode == 'changed':
                 df_editor_view = df_editor_view[df_editor_view['Visit_ID_Internal'].isin(changed_ids)]
                 if df_editor_view.empty: st.info("Chưa có KH nào bị thay đổi.")

            edited_df = st.data_editor(
                df_editor_view,
                column_config={
                    "Customer code": st.column_config.TextColumn("Mã KH", disabled=True),
                    "Customer Name": st.column_config.TextColumn("Tên KH", disabled=True),
                    "Frequency": st.column_config.NumberColumn("Tần suất", disabled=True, width="small"),
                    "Segment": st.column_config.TextColumn("Phân loại", disabled=True, width="small"),
                    "Week": st.column_config.SelectboxColumn("Tuần", options=['W1','W2','W3','W4'], required=True),
                    "Day": st.column_config.SelectboxColumn("Ngày", options=['T2','T3','T4','T5','T6','T7'], required=True),
                    "Sequence": st.column_config.NumberColumn("Thứ tự", disabled=True),
                    "Trạng thái": st.column_config.TextColumn("Trạng thái", disabled=True, width="small"),
                },
                column_order=['Trạng thái', 'Customer code', 'Customer Name', 'Frequency', 'Segment', 'Week', 'Day', 'Sequence'],
                hide_index=True, use_container_width=True, height=400, key="data_editor_widget"
            )
            
            # Control Logic
            has_unsaved_changes_vp = False
            if not edited_df['Week'].equals(df_editor_view['Week']) or not edited_df['Day'].equals(df_editor_view['Day']):
                has_unsaved_changes_vp = True
            
            # --- NEW LAYOUT: 1 Row for 3 main buttons [1, 1.2, 0.8] ---
            c_up, c_filter, c_clear = st.columns([1, 1.2, 0.8])
            
            with c_up:
                if st.button("💾 Cập nhật", type="primary", use_container_width=True):
                    with st.spinner("Đang tính toán lại lộ trình..."):
                        impacted_groups = set()
                        for idx, row in edited_df.iterrows():
                            if idx in df_editor_view.index:
                                visit_id = df_editor_view.loc[idx, 'Visit_ID_Internal']
                                mask = st.session_state.df_editing['Visit_ID_Internal'] == visit_id
                                if mask.any():
                                    current_row = st.session_state.df_editing.loc[mask].iloc[0]
                                    old_r, old_w, old_d = current_row['RouteID'], current_row['Week'], current_row['Day']
                                    new_w, new_d = row['Week'], row['Day']
                                    if (old_w != new_w) or (old_d != new_d):
                                        impacted_groups.add((old_r, old_w, old_d))
                                        impacted_groups.add((old_r, new_w, new_d))
                                        st.session_state.df_editing.loc[mask, ['Week', 'Day']] = [new_w, new_d]
                        if impacted_groups:
                            st.session_state.df_editing = recalculate_routes(st.session_state.df_editing, st.session_state.depot_coords, st.session_state.route_cfg, st.session_state.speed_cfg, impacted_groups=impacted_groups)
                            st.session_state.map_version += 1
                            st.session_state.has_changes = True 
                            st.session_state.vp_confirm_clear = False 
                            st.success("Đã cập nhật!")
                            time.sleep(0.5) 
                            st.rerun()
                        else: st.info("Không có thay đổi về Ngày/Tuần để cập nhật.")

            with c_filter:
                if st.button("🌪️ Lọc KH đã sửa", use_container_width=True, disabled=not st.session_state.has_changes):
                    st.session_state.editor_filter_mode = 'changed'
                    st.rerun()
            
            with c_clear:
                is_filtering_vp = (st.session_state.editor_filter_mode != 'all') or (st.session_state.map_clicked_code is not None)
                if st.button("✖ Bỏ lọc", use_container_width=True, disabled=not is_filtering_vp):
                    if has_unsaved_changes_vp:
                        st.session_state.vp_confirm_clear = True
                        st.rerun()
                    else:
                        st.session_state.editor_filter_mode = 'all'
                        st.session_state.map_clicked_code = None
                        st.session_state.map_version += 1 
                        st.rerun()
            
            # --- WARNING AREA BELOW MAIN BUTTONS ---
            if st.session_state.vp_confirm_clear:
                st.caption("⚠️ Bạn có thay đổi chưa lưu.")
                c_save_clear, c_discard_clear = st.columns(2)
                
                # Button: Save & Clear
                if c_save_clear.button("Lưu & Bỏ lọc", type="secondary", use_container_width=True):
                    with st.spinner("Đang tính toán lại lộ trình..."):
                        impacted_groups = set()
                        for idx, row in edited_df.iterrows():
                            if idx in df_editor_view.index:
                                visit_id = df_editor_view.loc[idx, 'Visit_ID_Internal']
                                mask = st.session_state.df_editing['Visit_ID_Internal'] == visit_id
                                if mask.any():
                                    current_row = st.session_state.df_editing.loc[mask].iloc[0]
                                    old_r, old_w, old_d = current_row['RouteID'], current_row['Week'], current_row['Day']
                                    new_w, new_d = row['Week'], row['Day']
                                    if (old_w != new_w) or (old_d != new_d):
                                        impacted_groups.add((old_r, old_w, old_d))
                                        impacted_groups.add((old_r, new_w, new_d))
                                        st.session_state.df_editing.loc[mask, ['Week', 'Day']] = [new_w, new_d]
                        if impacted_groups:
                            st.session_state.df_editing = recalculate_routes(st.session_state.df_editing, st.session_state.depot_coords, st.session_state.route_cfg, st.session_state.speed_cfg, impacted_groups=impacted_groups)
                            st.session_state.map_version += 1
                            st.session_state.has_changes = True 
                        
                        # Clear Logic
                        st.session_state.editor_filter_mode = 'all'
                        st.session_state.map_clicked_code = None
                        st.session_state.vp_confirm_clear = False
                        st.rerun()
                
                # Button: Discard & Clear
                if c_discard_clear.button("Không lưu & Bỏ lọc", type="secondary", use_container_width=True):
                    st.session_state.editor_filter_mode = 'all'
                    st.session_state.map_clicked_code = None
                    st.session_state.vp_confirm_clear = False
                    st.session_state.map_version += 1 
                    st.rerun()

            st.divider()
            
            if not st.session_state.confirm_reset:
                if st.button("🔄 Hủy bỏ & Reset", type="secondary", use_container_width=True, disabled=not st.session_state.has_changes):
                    st.session_state.confirm_reset = True
                    st.rerun()
            else:
                st.warning("Quay về phiên bản trước khi chỉnh sửa?")
                c_yes, c_no = st.columns(2)
                if c_yes.button("✅ Đồng ý", use_container_width=True):
                    st.session_state.df_editing = st.session_state.df_final.copy()
                    st.session_state.editor_filter_mode = 'all'
                    st.session_state.map_version += 1
                    st.session_state.has_changes = False
                    st.session_state.confirm_reset = False
                    st.rerun()
                if c_no.button("❌ Không", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()

        st.markdown("---")
        
        # --- EXPORT BUTTONS ---
        if is_integrated:
            if st.button("📥 Tải về (Tùy chọn)", type="primary"):
                st.session_state.show_download_options = True
            
            if st.session_state.show_download_options:
                c_d1, c_d2 = st.columns(2)
                excel_data = to_excel_output(st.session_state.df_editing)
                c_d1.download_button("File Xếp lịch viếng thăm", excel_data, "Result_Xeplichviengtham.xlsx")
                
                tp_mode = st.session_state.global_state['config']['tp_mode']
                tp_slug = "chedo1" if tp_mode == "Chế độ 1" else "chedo2"
                df_tp_res = st.session_state.v1_df_edited if tp_mode == "Chế độ 1" else st.session_state.v2_df_edited
                
                if df_tp_res is not None:
                    tp_buffer = to_excel_tp(df_tp_res)
                    c_d2.download_button(f"File Chia địa bàn ({tp_mode})", tp_buffer, f"Result_Chiadiaban_{tp_slug}.xlsx")
        else:
            excel_data = to_excel_output(st.session_state.df_editing)
            st.download_button("📥 Tải về", excel_data, "Result_Xeplichviengtham.xlsx", type='primary')

        if st.button("<< Quay lại từ đầu"):
            st.session_state.global_state['has_started'] = False
            st.session_state.global_state['step'] = 'welcome'
            st.rerun()

# ==========================================
# 5. MAIN CONTROLLER
# ==========================================

def main():
    render_sidebar()
    step = st.session_state.global_state['step']
    config = st.session_state.global_state['config']
    
    if not st.session_state.global_state['has_started']:
        render_welcome_screen()
    else:
        render_main_title()
        
        # --- SCENARIO A: TERRITORY PLANNER ONLY ---
        if config['is_tp'] and not config['is_integrated']:
            render_tp_ui(is_integrated=False)
        
        # --- SCENARIO B: VISIT PLANNER ONLY ---
        elif config['is_vp'] and not config['is_integrated']:
            render_vp_ui(is_integrated=False)
                
        # --- SCENARIO C: INTEGRATED (SUPER APP) ---
        elif config['is_integrated']:
            # Input Stage (VP Style)
            if step == 'input_integrated':
                render_vp_ui(is_integrated=True)
            # TP Process Stage (TP UI)
            elif step == 'tp_setup' or (step == 'tp_process' and st.session_state.page in ['setup', 'result']):
                render_tp_ui(is_integrated=True)
            elif step == 'tp_result_integrated':
                render_tp_ui(is_integrated=True) # Will render result part
            # VP Process Stage (VP UI)
            elif step in ['vp_process', 'vp_result']:
                render_vp_ui(is_integrated=True)

if __name__ == "__main__":
    main()