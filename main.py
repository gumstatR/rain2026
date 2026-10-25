import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
import streamlit as st
import matplotlib_fontja

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="愛知県雨量比較Webアプリ", layout="wide")
st.title("愛知県における雨量比較Webアプリ")
st.write("時系列グラフと最大雨量分布（カラー面塗りマップ）を用いて2000年9月東海豪雨と2026年9月豪雨を比較することができます。")

# --- 1. データロード＆前処理（キャッシュ管理） ---
@st.cache_data
def load_all_data():
    df_2000 = pd.read_csv("200009.csv", encoding="utf-8")
    df_2026 = pd.read_csv("202609.csv", encoding="utf-8")
    
    # 列名の空白除去＆日時列の共通化
    df_2000.columns = [c.strip() for c in df_2000.columns]
    df_2026.columns = [c.strip() for c in df_2026.columns]
    
    df_2000.rename(columns={df_2000.columns[0]: "DateTime"}, inplace=True)
    df_2026.rename(columns={df_2026.columns[0]: "DateTime"}, inplace=True)
    
    # 日時フォーマット変換
    df_2000["DateTime"] = pd.to_datetime(df_2000["DateTime"].str.replace("年", "/").str.replace("月", "/").str.replace("日", "").str.replace("時", ":").str.replace("分", ""), errors="coerce")
    df_2026["DateTime"] = pd.to_datetime(df_2026["DateTime"].str.replace("年", "/").str.replace("月", "/").str.replace("日", "").str.replace("時", ":").str.replace("分", ""), errors="coerce")
    
    # 位置情報 CSV (location.csv) の読み込み
    df_loc = pd.read_csv("location.csv", encoding="utf-8")
    df_loc["observatory"] = df_loc["observatory"].astype(str).str.strip()
    
    # area が欠損している行や "nan" のものを安全に除外
    df_loc = df_loc.dropna(subset=["area"])
    df_loc["area"] = df_loc["area"].astype(str).str.strip()
    df_loc = df_loc[~df_loc["area"].str.lower().isin(["nan", ""])]
    
    stations = df_loc[["observatory", "latitude", "longitude", "area"]].rename(
        columns={"observatory": "観測所名", "latitude": "lat", "longitude": "lon", "area": "エリア"}
    ).copy()

    # 集計データフレームの作成（高速化・フラグメント防止）
    def calc_rain_stats(df):
        cols = [c for c in df.columns if c != "DateTime"]
        numeric_df = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        cum_df = numeric_df.cumsum()
        cum_df.insert(0, "DateTime", df["DateTime"])
        
        r6h_df = numeric_df.rolling(window=6, min_periods=1).sum()
        r6h_df.insert(0, "DateTime", df["DateTime"])
        
        r12h_df = numeric_df.rolling(window=12, min_periods=1).sum()
        r12h_df.insert(0, "DateTime", df["DateTime"])
        
        clean_df = numeric_df.copy()
        clean_df.insert(0, "DateTime", df["DateTime"])
        
        return clean_df, cum_df, r6h_df, r12h_df

    df_2000_c, cum_2000, r6h_2000, r12h_2000 = calc_rain_stats(df_2000)
    df_2026_c, cum_2026, r6h_2026, r12h_2026 = calc_rain_stats(df_2026)
    
    return {
        "2000": {"1h": df_2000_c, "cum": cum_2000, "6h": r6h_2000, "12h": r12h_2000},
        "2026": {"1h": df_2026_c, "cum": cum_2026, "6h": r6h_2026, "12h": r12h_2026},
        "stations": stations
    }

@st.cache_data
def load_geojson_and_mask(geojson_path="aichiboundary.geojson"):
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        paths = []
        polygons_coords = []
        for feature in data.get("features", []):
            geom = feature.get("geometry", {})
            g_type = geom.get("type")
            coords = geom.get("coordinates")
            
            rings = []
            if g_type == "Polygon":
                rings = coords
            elif g_type == "MultiPolygon":
                rings = [ring for poly in coords for ring in poly]
                
            for ring in rings:
                poly_verts = [(pt[0], pt[1]) for pt in ring]
                paths.append(Path(poly_verts))
                polygons_coords.append(poly_verts)
                
        grid_lon = np.linspace(136.5, 137.9, 100)
        grid_lat = np.linspace(34.5, 35.45, 100)
        GRID_LON, GRID_LAT = np.meshgrid(grid_lon, grid_lat)
        
        grid_points = np.vstack((GRID_LON.flatten(), GRID_LAT.flatten())).T
        mask = np.ones(len(grid_points), dtype=bool)
        for p in paths:
            mask = mask & (~p.contains_points(grid_points))
        mask_reshaped = mask.reshape(GRID_LON.shape)
        
        return paths, polygons_coords, GRID_LON, GRID_LAT, mask_reshaped
    except Exception as e:
        st.warning(f"GeoJSON読み込みスキップ (境界線描画なし): {e}")
        return [], [], None, None, None

data_store = load_all_data()
stations = data_store["stations"]
geojson_paths, polygons_coords, GRID_LON, GRID_LAT, MASK_GRID = load_geojson_and_mask()

# --- 2. 時系列グラフ描画ヘルパー ---
def create_timeseries_plot(df_1h, df_cum, loc, title, ymax_1h, ymax_cum):
    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    
    p1 = ax1.plot(df_1h["DateTime"], df_1h[loc], label="1時間雨量", color="tab:blue", alpha=0.8)
    ax1.set_xlabel("日時", fontsize=9)
    ax1.set_ylabel("1時間雨量 (mm)", color="tab:blue", fontsize=9)
    ax1.set_ylim(0, ymax_1h)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=25, ha='right', fontsize=9)
    
    ax2 = ax1.twinx()
    p2 = ax2.plot(df_cum["DateTime"], df_cum[loc], label="累積雨量", color="tab:orange", linestyle="--", alpha=0.8)
    ax2.set_ylabel("累積雨量 (mm)", color="tab:orange", fontsize=9)
    ax2.set_ylim(0, ymax_cum)
    ax2.grid(False)
    
    plots = p1 + p2
    labels = [l.get_label() for l in plots]
    ax1.legend(plots, labels, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig

# --- 3. マップ描画ヘルパー（カラーバー目盛りをキリの良い数字に整理） ---
def create_rain_map(row_data, event_title, mode, global_max_limit):
    fig, ax = plt.subplots(figsize=(7, 5.8))

    for poly_verts in polygons_coords:
        xs = [pt[0] for pt in poly_verts]
        ys = [pt[1] for pt in poly_verts]
        ax.plot(xs, ys, color="darkgreen", linewidth=0.8, zorder=4)

    if not row_data.empty:
        rain_melted = row_data.T.reset_index()
        rain_melted.columns = ["観測所名", "雨量"]
        map_df = pd.merge(rain_melted, stations, on="観測所名", how="inner")
        
        map_df["雨量"] = pd.to_numeric(map_df["雨量"], errors="coerce").fillna(0)
        map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
        map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
        map_df = map_df.dropna(subset=["lat", "lon", "雨量"])

        mode_config = {
            "12時間雨量": ("Purples", "12時間雨量 (mm)"),
            "6時間雨量": ("Greens", "6時間雨量 (mm)"),
            "1時間雨量": ("Blues", "1時間雨量 (mm)")
        }
        cmap_name, unit_label = mode_config[mode]
        
        levels_f = np.linspace(0, global_max_limit, 16)

        if len(map_df) > 3 and MASK_GRID is not None:
            points = map_df[["lon", "lat"]].values
            values = map_df["雨量"].values
            
            valid_idx = ~np.isnan(points).any(axis=1) & ~np.isnan(values)
            points = points[valid_idx]
            values = values[valid_idx]

            if len(points) > 3:
                GRID_RAIN_linear = griddata(points, values, (GRID_LON, GRID_LAT), method='linear')
                GRID_RAIN_nearest = griddata(points, values, (GRID_LON, GRID_LAT), method='nearest')
                GRID_RAIN = np.where(np.isnan(GRID_RAIN_linear), GRID_RAIN_nearest, GRID_RAIN_linear)
                
                GRID_RAIN_masked = np.ma.array(GRID_RAIN, mask=MASK_GRID)

                contourf = ax.contourf(
                    GRID_LON, GRID_LAT, GRID_RAIN_masked, 
                    levels=levels_f, cmap=cmap_name, alpha=0.8, zorder=2, 
                    vmin=0, vmax=global_max_limit
                )

                cbar = fig.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(unit_label)
                # カラーバーの目盛りをキリの良い整数の間隔に自動調整
                cbar.locator = ticker.MaxNLocator(nbins=6, integer=True)
                cbar.update_ticks()

        ax.scatter(map_df["lon"], map_df["lat"], color="red", s=15, zorder=5, label="観測地点")

        for _, row in map_df.iterrows():
            ax.text(
                row["lon"] + 0.01, 
                row["lat"] + 0.005, 
                str(row["観測所名"]), 
                fontsize=6, 
                color="red", 
                weight="bold",
                zorder=10
            )

    ax.set_xlim(136.55, 137.90)
    ax.set_ylim(34.50, 35.45)
    ax.set_xlabel("経度", fontsize=9)
    ax.set_ylabel("緯度", fontsize=9)
    
    ax.set_title(f"{event_title} [{mode}]", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    return fig

# --- 4. メインUI表示 ---
st.subheader("📈 観測地点の時系列グラフ比較")

valid_stations_list = stations["観測所名"].tolist()
common_locations = [c for c in data_store["2026"]["1h"].columns if c != "DateTime" and c in data_store["2000"]["1h"].columns and c in valid_stations_list]

station_to_area = dict(zip(stations["観測所名"], stations["エリア"]))
areas = sorted(list(stations["エリア"].dropna().unique()))

col_a1, col_a2 = st.columns(2)
with col_a1:
    selected_area = st.selectbox("エリアを選択", areas, index=areas.index("尾張東部") if "尾張東部" in areas else 0)

filtered_locations = [loc for loc in common_locations if station_to_area.get(loc) == selected_area]
if not filtered_locations:
    filtered_locations = common_locations

with col_a2:
    selected_loc = st.selectbox("雨量観測局を選択", filtered_locations)

max_1h = max(data_store["2000"]["1h"][selected_loc].max(), data_store["2026"]["1h"][selected_loc].max(), 10.0) * 1.1
max_cum = max(data_store["2000"]["cum"][selected_loc].max(), data_store["2026"]["cum"][selected_loc].max(), 50.0) * 1.1

# 📊 選択地点の統計情報表示
st.markdown(f"**📍 観測局 [{selected_loc}] の雨量情報**")

st.markdown("##### 2000年 東海豪雨")
m1, m2, m3, m4 = st.columns(4)
m1.metric("1時間雨量", f"{data_store['2000']['1h'][selected_loc].max():.1f} mm")
m2.metric("6時間雨量", f"{data_store['2000']['6h'][selected_loc].max():.1f} mm")
m3.metric("12時間雨量", f"{data_store['2000']['12h'][selected_loc].max():.1f} mm")
m4.metric("総累積雨量", f"{data_store['2000']['cum'][selected_loc].max():.1f} mm")

st.markdown("##### 2026年 9月豪雨")
m5, m6, m7, m8 = st.columns(4)
m5.metric("1時間雨量", f"{data_store['2026']['1h'][selected_loc].max():.1f} mm")
m6.metric("6時間雨量", f"{data_store['2026']['6h'][selected_loc].max():.1f} mm")
m7.metric("12時間雨量", f"{data_store['2026']['12h'][selected_loc].max():.1f} mm")
m8.metric("総累積雨量", f"{data_store['2026']['cum'][selected_loc].max():.1f} mm")

col3, col4 = st.columns(2)
with col3:
    st.markdown(f"#### 2000年 東海豪雨 ({selected_loc})")
    fig_2000 = create_timeseries_plot(data_store["2000"]["1h"], data_store["2000"]["cum"], selected_loc, "2000年", max_1h, max_cum)
    st.pyplot(fig_2000, clear_figure=True)
    plt.close(fig_2000)

with col4:
    st.markdown(f"#### 2026年 9月豪雨 ({selected_loc})")
    fig_2026 = create_timeseries_plot(data_store["2026"]["1h"], data_store["2026"]["cum"], selected_loc, "2026年", max_1h, max_cum)
    st.pyplot(fig_2026, clear_figure=True)
    plt.close(fig_2026)

st.markdown("---")

st.subheader("☔️ 各最大雨量分布（カラー面塗りマップ）")
map_mode = st.radio("表示する時間を選択", ["1時間雨量", "6時間雨量", "12時間雨量"], horizontal=True)

mode_key_map = {"1時間雨量": "1h", "6時間雨量": "6h", "12時間雨量": "12h"}
current_key = mode_key_map[map_mode]

def get_max_val(year):
    tdf = data_store[year][current_key]
    num_cols = [c for c in tdf.columns if c != "DateTime"]
    return tdf[num_cols].apply(pd.to_numeric, errors='coerce').max().max()

# --- キリの良い数字に自動切り上げ（スマート丸め関数） ---
def nice_ceil(val):
    if val <= 20:
        return np.ceil(val / 5) * 5
    elif val <= 100:
        return np.ceil(val / 10) * 10
    else:
        return np.ceil(val / 50) * 50

raw_max = max(get_max_val("2000"), get_max_val("2026"))
global_max_limit = nice_ceil(raw_max)
if global_max_limit == 0:
    global_max_limit = 20.0  # 念のためのフォールバック

def render_max_rain_map(year, title, col):
    with col:
        st.markdown(f"#### {title}")
        target_df = data_store[year][current_key]
        
        numeric_cols = [c for c in target_df.columns if c != "DateTime"]
        max_series = target_df[numeric_cols].apply(pd.to_numeric, errors='coerce').max()
        row_data = max_series.to_frame().T
        
        fig_map = create_rain_map(row_data, f"{title} (期間内最大値)", map_mode, global_max_limit)
        st.pyplot(fig_map, clear_figure=True)
        plt.close(fig_map)

map_col1, map_col2 = st.columns(2)
render_max_rain_map("2000", "2000年 東海豪雨", map_col1)
render_max_rain_map("2026", "2026年 9月豪雨", map_col2)