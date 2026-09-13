import os
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.path import Path
from scipy.interpolate import griddata
import streamlit as st
import japanize_matplotlib


matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="愛知県雨量比較Webアプリ", layout="wide")
st.title("愛知県における東海豪雨と2026年雨量比較Webアプリ")
st.write("2000年9月東海豪雨と2026年9月豪雨の時系列グラフおよび等雨量線図を比較します。")

# --- 1. データロード＆前処理（キャッシュ管理） ---
@st.cache_data
def load_all_data():
    df_2000 = pd.read_csv("2000raindata.csv", encoding="utf-8")
    df_2026 = pd.read_csv("2026raindata.csv", encoding="utf-8")
    
    df_2000.rename(columns={df_2000.columns[0]: "DateTime"}, inplace=True)
    df_2026.rename(columns={df_2026.columns[0]: "DateTime"}, inplace=True)
    
    df_2000["DateTime"] = pd.to_datetime(df_2000["DateTime"], errors="coerce")
    df_2026["DateTime"] = pd.to_datetime(df_2026["DateTime"], errors="coerce")
    
    df_master = pd.read_csv("ame_master_20260324.csv", encoding="cp932")
    aichi_master = df_master[df_master["都府県振興局"] == "愛知"].copy()
    aichi_master["lat"] = aichi_master["緯度(度)"] + aichi_master["緯度(分)"] / 60.0
    aichi_master["lon"] = aichi_master["経度(度)"] + aichi_master["経度(分)"] / 60.0
    stations = aichi_master[["観測所名", "lat", "lon"]].copy()

    # 集計データフレームの作成
    def calc_rain_stats(df):
        cols = [c for c in df.columns if c != "DateTime"]
        numeric_df = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        cum = df.copy()
        cum[cols] = numeric_df.cumsum()
        
        r6h = df.copy()
        r6h[cols] = numeric_df.rolling(window=6, min_periods=1).sum()
        
        r12h = df.copy()
        r12h[cols] = numeric_df.rolling(window=12, min_periods=1).sum()
        
        return cum, r6h, r12h

    cum_2000, r6h_2000, r12h_2000 = calc_rain_stats(df_2000)
    cum_2026, r6h_2026, r12h_2026 = calc_rain_stats(df_2026)
    
    return {
        "2000": {"1h": df_2000, "cum": cum_2000, "6h": r6h_2000, "12h": r12h_2000},
        "2026": {"1h": df_2026, "cum": cum_2026, "6h": r6h_2026, "12h": r12h_2026},
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
                
        # メッシュグリッドとマスクの事前計算（重い処理のキャッシュ化）
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
        st.error(f"GeoJSON読み込みエラー: {e}")
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

# --- 3. 等雨量線マップ描画ヘルパー ---
def create_rain_map(row_data, event_title, display_time, mode):
    fig, ax = plt.subplots(figsize=(7, 5.8))

    for poly_verts in polygons_coords:
        xs = [pt[0] for pt in poly_verts]
        ys = [pt[1] for pt in poly_verts]
        ax.plot(xs, ys, color="darkgreen", linewidth=1.8, zorder=4)

    if not row_data.empty:
        rain_melted = row_data.T.reset_index()
        rain_melted.columns = ["観測所名", "雨量"]
        map_df = pd.merge(rain_melted, stations, on="観測所名", how="inner")
        map_df["雨量"] = pd.to_numeric(map_df["雨量"], errors="coerce").fillna(0)

        # マップ色・上限の判定
        mode_config = {
            "12時間雨量": ("Purples", "12時間雨量 (mm)", 100.0, "12h"),
            "6時間雨量": ("Greens", "6時間雨量 (mm)", 80.0, "6h"),
            "1時間雨量": ("Blues", "1時間雨量 (mm)", 50.0, "1h")
        }
        cmap_name, unit_label, base_limit, data_key = mode_config[mode]
        
        max_2000 = data_store["2000"][data_key].drop(columns=["DateTime"], errors="ignore").apply(pd.to_numeric, errors='coerce').max().max()
        max_2026 = data_store["2026"][data_key].drop(columns=["DateTime"], errors="ignore").apply(pd.to_numeric, errors='coerce').max().max()
        max_rain_limit = max(max_2000, max_2026, base_limit)

        levels_f = np.linspace(0, max_rain_limit, 15)
        levels_c = np.linspace(0, max_rain_limit, 8)

        if len(map_df) > 3 and MASK_GRID is not None:
            points = map_df[["lon", "lat"]].values
            values = map_df["雨量"].values
            GRID_RAIN = griddata(points, values, (GRID_LON, GRID_LAT), method='linear')
            GRID_RAIN_masked = np.ma.array(GRID_RAIN, mask=MASK_GRID)

            contourf = ax.contourf(GRID_LON, GRID_LAT, GRID_RAIN_masked, levels=levels_f, cmap=cmap_name, alpha=0.8, zorder=2, vmin=0, vmax=max_rain_limit)
            contour = ax.contour(GRID_LON, GRID_LAT, GRID_RAIN_masked, levels=levels_c, colors="navy", linewidths=0.5, zorder=3, vmin=0, vmax=max_rain_limit)
            ax.clabel(contour, inline=True, fontsize=7, fmt="%.0f mm")

            cbar = fig.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit_label)

        ax.scatter(map_df["lon"], map_df["lat"], color="red", s=25, zorder=5, label="観測地点")
        for _, row in map_df.iterrows():
            ax.text(row["lon"] + 0.015, row["lat"], f"{row['観測所名']} ({row['雨量']:.1f})", fontsize=7, color="black", zorder=6)

    ax.set_xlim(136.55, 137.90)
    ax.set_ylim(34.50, 35.45)
    ax.set_xlabel("経度", fontsize=9)
    ax.set_ylabel("緯度", fontsize=9)
    
    time_str = display_time.strftime('%Y-%m-%d %H:%M') if isinstance(display_time, pd.Timestamp) else str(display_time)
    ax.set_title(f"{event_title} [{mode}] ({time_str})", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    return fig

# --- 4. メインUI表示 ---
st.subheader("📈 観測地点の時系列グラフ比較")

common_locations = [c for c in data_store["2026"]["1h"].columns if c != "DateTime" and c in data_store["2000"]["1h"].columns]
selected_loc = st.selectbox("観測地点を選択", common_locations, index=common_locations.index("名古屋") if "名古屋" in common_locations else 0)

# Y軸共通範囲の算出
max_1h = max(data_store["2000"]["1h"][selected_loc].max(), data_store["2026"]["1h"][selected_loc].max(), 10.0) * 1.1
max_cum = max(data_store["2000"]["cum"][selected_loc].max(), data_store["2026"]["cum"][selected_loc].max(), 50.0) * 1.1

# 概要メトリクスカード表示
m1, m2, m3, m4 = st.columns(4)
m1.metric("2000年 最大1時間雨量", f"{data_store['2000']['1h'][selected_loc].max():.1f} mm")
m2.metric("2000年 総累積雨量", f"{data_store['2000']['cum'][selected_loc].max():.1f} mm")
m3.metric("2026年 最大1時間雨量", f"{data_store['2026']['1h'][selected_loc].max():.1f} mm")
m4.metric("2026年 総累積雨量", f"{data_store['2026']['cum'][selected_loc].max():.1f} mm")

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

st.subheader("☔️ 等雨量線図の比較")
map_mode = st.radio("表示する雨量タイプを選択", ["1時間雨量", "6時間雨量", "12時間雨量"], horizontal=True)

mode_key_map = {"1時間雨量": "1h", "6時間雨量": "6h", "12時間雨量": "12h"}
current_key = mode_key_map[map_mode]

# 時間選択ボックスのヘルパー関数
def render_map_section(year, title, col):
    with col:
        st.markdown(f"#### {title}")
        target_df = data_store[year][current_key]
        time_list = target_df["DateTime"].tolist()
        time_labels = [t.strftime('%Y-%m-%d %H:%M') if isinstance(t, pd.Timestamp) else str(t) for t in time_list]
        
        # 初期選択をピーク日時にあわせる
        state_key = f"time_{year}_{selected_loc}_{map_mode}"
        if state_key not in st.session_state or st.session_state[state_key] not in time_labels:
            peak_idx = target_df[selected_loc].idxmax() if selected_loc in target_df.columns else 0
            st.session_state[state_key] = time_labels[peak_idx]

        selected_label = st.selectbox(f"時間選択 ({year}年)", time_labels, key=state_key)
        selected_time = time_list[time_labels.index(selected_label)]
        
        row_data = target_df[target_df["DateTime"] == selected_time].drop(columns=["DateTime"], errors="ignore")
        fig_map = create_rain_map(row_data, title, selected_time, map_mode)
        st.pyplot(fig_map, clear_figure=True)
        plt.close(fig_map)

map_col1, map_col2 = st.columns(2)
render_map_section("2000", "2000年 東海豪雨", map_col1)
render_map_section("2026", "2026年 9月豪雨", map_col2)
# #おまけ
# #streamlit run main.pyでローカルで確認できる。
# #初期設定①git init, ②git remote add origin https:/// 
# #　①git add . ②git commit -m"◯◯◯" .　③git push origin main
