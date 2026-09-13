import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from scipy.interpolate import griddata
from matplotlib.path import Path
import json

# 日本語フォント対応
import platform
if platform.system() == 'Darwin':
    matplotlib.rcParams['font.family'] = 'Hiragino Sans'  # Mac用
else:
    matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'  # Linux (Streamlit Cloud) 用

matplotlib.rcParams['axes.unicode_minus'] = False

st.title("愛知県における東海豪雨と2026年雨量比較Webアプリ")
st.write("2000年9月東海豪雨と2026年9月豪雨の時系列グラフおよび等雨量線図を比較します。")

# データのロード関数
@st.cache_data
def load_all_data():
    df_2000 = pd.read_csv("2000raindata.csv", encoding="utf-8")
    df_2026 = pd.read_csv("2026raindata.csv", encoding="utf-8")
    
    df_2000.rename(columns={df_2000.columns[0]: "DateTime"}, inplace=True)
    df_2026.rename(columns={df_2026.columns[0]: "DateTime"}, inplace=True)
    
    df_2000 = df_2000.reset_index(drop=True)
    df_2026 = df_2026.reset_index(drop=True)
    
    df_2000["DateTime"] = pd.to_datetime(df_2000["DateTime"], errors="coerce")
    df_2026["DateTime"] = pd.to_datetime(df_2026["DateTime"], errors="coerce")
    
    df_master = pd.read_csv("ame_master_20260324.csv", encoding="cp932")
    aichi_master = df_master[df_master["都府県振興局"] == "愛知"].copy()
    
    aichi_master["lat"] = aichi_master["緯度(度)"] + aichi_master["緯度(分)"] / 60.0
    aichi_master["lon"] = aichi_master["経度(度)"] + aichi_master["経度(分)"] / 60.0
    
    stations = aichi_master[["観測所名", "lat", "lon"]].copy()
    
    return df_2000, df_2026, stations

@st.cache_data
def load_geojson_paths(geojson_path="aichiboundary.geojson"):
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    paths = []
    polygons_coords = []
    
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        g_type = geom.get("type")
        coords = geom.get("coordinates")
        
        if g_type == "Polygon":
            for ring in coords:
                poly_verts = [(pt[0], pt[1]) for pt in ring]
                paths.append(Path(poly_verts))
                polygons_coords.append(poly_verts)
        elif g_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    poly_verts = [(pt[0], pt[1]) for pt in ring]
                    paths.append(Path(poly_verts))
                    polygons_coords.append(poly_verts)
                    
    return paths, polygons_coords

df_2000, df_2026, stations = load_all_data()

# --- 累積雨量データフレームの作成 ---
df_2000_cum = df_2000.copy()
numeric_cols_2000 = [c for c in df_2000.columns if c != "DateTime"]
df_2000_cum[numeric_cols_2000] = df_2000[numeric_cols_2000].apply(pd.to_numeric, errors='coerce').fillna(0).cumsum()

df_2026_cum = df_2026.copy()
numeric_cols_2026 = [c for c in df_2026.columns if c != "DateTime"]
df_2026_cum[numeric_cols_2026] = df_2026[numeric_cols_2026].apply(pd.to_numeric, errors='coerce').fillna(0).cumsum()

# --- 6時間・12時間移動積算雨量データフレームの作成 ---
df_2000_6h = df_2000.copy()
df_2000_6h[numeric_cols_2000] = df_2000[numeric_cols_2000].apply(pd.to_numeric, errors='coerce').fillna(0).rolling(window=6, min_periods=1).sum()

df_2026_6h = df_2026.copy()
df_2026_6h[numeric_cols_2026] = df_2026[numeric_cols_2026].apply(pd.to_numeric, errors='coerce').fillna(0).rolling(window=6, min_periods=1).sum()

df_2000_12h = df_2000.copy()
df_2000_12h[numeric_cols_2000] = df_2000[numeric_cols_2000].apply(pd.to_numeric, errors='coerce').fillna(0).rolling(window=12, min_periods=1).sum()

df_2026_12h = df_2026.copy()
df_2026_12h[numeric_cols_2026] = df_2026[numeric_cols_2026].apply(pd.to_numeric, errors='coerce').fillna(0).rolling(window=12, min_periods=1).sum()

try:
    geojson_paths, polygons_coords = load_geojson_paths("aichiboundary.geojson")
except Exception as e:
    st.error(f"GeoJSONファイルの読み込みに失敗しました: {e}")
    geojson_paths, polygons_coords = [], []


# マップを描画する共通関数
def create_rain_map(row_data, event_title, display_time, mode):
    fig, ax = plt.subplots(figsize=(7, 6))

    for poly_verts in polygons_coords:
        xs = [pt[0] for pt in poly_verts]
        ys = [pt[1] for pt in poly_verts]
        ax.plot(xs, ys, color="darkgreen", linewidth=2, zorder=4)

    if not row_data.empty:
        rain_melted = row_data.T.reset_index()
        rain_melted.columns = ["観測所名", "雨量"]

        map_df = pd.merge(rain_melted, stations, on="観測所名", how="inner")
        map_df["雨量"] = pd.to_numeric(map_df["雨量"], errors="coerce").fillna(0)

        if mode == "12時間雨量":
            max_rain_limit = max(df_2000_12h[numeric_cols_2000].max().max(), df_2026_12h[numeric_cols_2026].max().max(), 100.0)
            cmap_name = "Purples"
            unit_label = "12時間雨量 (mm)"
        elif mode == "6時間雨量":
            max_rain_limit = max(df_2000_6h[numeric_cols_2000].max().max(), df_2026_6h[numeric_cols_2026].max().max(), 80.0)
            cmap_name = "Greens"
            unit_label = "6時間雨量 (mm)"
        else:
            max_rain_limit = max(df_2000[numeric_cols_2000].apply(pd.to_numeric, errors='coerce').max().max(), 
                                 df_2026[numeric_cols_2026].apply(pd.to_numeric, errors='coerce').max().max(), 50.0)
            cmap_name = "Blues"
            unit_label = "1時間雨量 (mm)"

        levels_f = np.linspace(0, max_rain_limit, 15)
        levels_c = np.linspace(0, max_rain_limit, 8)

        if len(map_df) > 3 and geojson_paths:
            points = map_df[["lon", "lat"]].values
            values = map_df["雨量"].values

            grid_lon = np.linspace(136.5, 137.9, 100)
            grid_lat = np.linspace(34.5, 35.45, 100)
            GRID_LON, GRID_LAT = np.meshgrid(grid_lon, grid_lat)

            GRID_RAIN = griddata(points, values, (GRID_LON, GRID_LAT), method='linear')

            grid_points = np.vstack((GRID_LON.flatten(), GRID_LAT.flatten())).T
            mask = np.ones(len(grid_points), dtype=bool)
            for p in geojson_paths:
                mask = mask & (~p.contains_points(grid_points))
                
            GRID_RAIN_masked = np.ma.array(GRID_RAIN, mask=mask.reshape(GRID_LON.shape))

            contourf = ax.contourf(GRID_LON, GRID_LAT, GRID_RAIN_masked, levels=levels_f, cmap=cmap_name, alpha=0.8, zorder=2, vmin=0, vmax=max_rain_limit)
            contour = ax.contour(GRID_LON, GRID_LAT, GRID_RAIN_masked, levels=levels_c, colors="navy", linewidths=0.5, zorder=3, vmin=0, vmax=max_rain_limit)
            ax.clabel(contour, inline=True, fontsize=7, fmt="%.0f mm")

            cbar = fig.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit_label)

        ax.scatter(map_df["lon"], map_df["lat"], color="red", s=25, zorder=5, label="観測地点")
        
        for _, row in map_df.iterrows():
            label_text = f"{row['観測所名']} ({row['雨量']:.1f})"
            ax.text(row["lon"] + 0.02, row["lat"], label_text, fontsize=7, color="black", zorder=6)

    ax.set_xlim(136.55, 137.90)
    ax.set_ylim(34.50, 35.45)
    ax.set_xlabel("経度", fontsize=10)
    ax.set_ylabel("緯度", fontsize=10)
    
    time_str = display_time.strftime('%Y-%m-%d %H:%M') if isinstance(display_time, pd.Timestamp) else str(display_time)
    ax.set_title(f"{event_title} [{mode}] ({time_str})", fontsize=12)
    ax.plot([], [], color="darkgreen", linewidth=2, label="愛知県 境界・海岸線")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    return fig


# --- 1. 上部：時系列グラフ ---
st.subheader("📈 観測地点の時系列グラフ比較 (1時間雨量、累積雨量)")

common_locations = [col for col in df_2026.columns if col != "DateTime" and col in df_2000.columns]
selected_loc = st.selectbox("グラフを表示する観測地点を選択", common_locations, index=common_locations.index("名古屋") if "名古屋" in common_locations else 0)

max_y_2000 = df_2000[selected_loc].max() if selected_loc in df_2000.columns else 0
max_y_2026 = df_2026[selected_loc].max() if selected_loc in df_2026.columns else 0
common_ymax = max(max_y_2000, max_y_2026, 10.0) * 1.1

max_cum_2000 = df_2000_cum[selected_loc].max() if selected_loc in df_2000_cum.columns else 0
max_cum_2026 = df_2026_cum[selected_loc].max() if selected_loc in df_2026_cum.columns else 0
common_ymax_cum = max(max_cum_2000, max_cum_2026, 50.0) * 1.1

col3, col4 = st.columns(2)

def format_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)

# 2000年側の時系列グラフ
with col3:
    st.markdown(f"#### 2000年 東海豪雨 ({selected_loc})")
    fig_g1, ax_g1 = plt.subplots(figsize=(6, 4))
    
    p1 = ax_g1.plot(df_2000["DateTime"], df_2000[selected_loc], label="1時間雨量", color="tab:blue", alpha=0.8)
    ax_g1.set_xlabel("日時", fontsize=9)
    ax_g1.set_ylabel("1時間雨量 (mm)", color="tab:blue", fontsize=9)
    ax_g1.set_ylim(0, common_ymax)
    ax_g1.grid(True, linestyle="--", alpha=0.5)
    
    ax_g1_right = ax_g1.twinx()
    p2 = ax_g1_right.plot(df_2000_cum["DateTime"], df_2000_cum[selected_loc], label="累積雨量", color="tab:orange", linestyle="--", alpha=0.8)
    ax_g1_right.set_ylabel("累積雨量 (mm)", color="tab:orange", fontsize=9)
    ax_g1_right.set_ylim(0, common_ymax_cum)
    ax_g1_right.grid(False)
    
    ps = p1 + p2
    labs = [l.get_label() for l in ps]
    ax_g1.legend(ps, labs, loc="upper left", fontsize=8)
    
    format_time_axis(ax_g1)
    fig_g1.tight_layout()
    st.pyplot(fig_g1, clear_figure=True)
    plt.close(fig_g1)

# 2026年側の時系列グラフ
with col4:
    st.markdown(f"#### 2026年 9月豪雨 ({selected_loc})")
    fig_g2, ax_g2 = plt.subplots(figsize=(6, 4))
    
    p1 = ax_g2.plot(df_2026["DateTime"], df_2026[selected_loc], label="1時間雨量", color="tab:blue", alpha=0.8)
    ax_g2.set_xlabel("日時", fontsize=9)
    ax_g2.set_ylabel("1時間雨量 (mm)", color="tab:blue", fontsize=9)
    ax_g2.set_ylim(0, common_ymax)
    ax_g2.grid(True, linestyle="--", alpha=0.5)
    
    ax_g2_right = ax_g2.twinx()
    p2 = ax_g2_right.plot(df_2026_cum["DateTime"], df_2026_cum[selected_loc], label="累積雨量", color="tab:orange", linestyle="--", alpha=0.8)
    ax_g2_right.set_ylabel("累積雨量 (mm)", color="tab:orange", fontsize=9)
    ax_g2_right.set_ylim(0, common_ymax_cum)
    ax_g2_right.grid(False)
    
    ps = p1 + p2
    labs = [l.get_label() for l in ps]
    ax_g2.legend(ps, labs, loc="upper left", fontsize=8)
    
    format_time_axis(ax_g2)
    fig_g2.tight_layout()
    st.pyplot(fig_g2, clear_figure=True)
    plt.close(fig_g2)

st.markdown("---")

# --- 2. 下部：等雨量線図（1時間 / 6時間 / 12時間 切替対応） ---
st.subheader("☔️ 等雨量線図の比較")

map_mode = st.radio("表示する等雨量線の種類を選択", ["1時間雨量", "6時間雨量", "12時間雨量"], horizontal=True)

# モードに応じて参照するデータフレームを切り替え
if map_mode == "12時間雨量":
    target_df_2000 = df_2000_12h
    target_df_2026 = df_2026_12h
elif map_mode == "6時間雨量":
    target_df_2000 = df_2000_6h
    target_df_2026 = df_2026_6h
else:
    target_df_2000 = df_2000
    target_df_2026 = df_2026

# 地点やモードが切り替わったときにデフォルトの選択状態を強制的にピーク（最大値）に同期させる処理
state_key_2000 = f"time_2000_{selected_loc}_{map_mode}"
state_key_2026 = f"time_2026_{selected_loc}_{map_mode}"

col1, col2 = st.columns(2)
# 2000年 東海豪雨エリア
with col1:
    st.markdown(f"####  2000年 東海豪雨")
    time_list_2000 = target_df_2000["DateTime"].tolist()
    time_labels_2000 = [t.strftime('%Y-%m-%d %H:%M') if isinstance(t, pd.Timestamp) else str(t) for t in time_list_2000]
    
    default_index_2000 = 0
    if selected_loc in target_df_2000.columns and not target_df_2000.empty:
        s_vals = pd.to_numeric(target_df_2000[selected_loc], errors='coerce')
        if not s_vals.isna().all():
            max_idx_2000 = s_vals.idxmax()
            peak_time_2000 = target_df_2000.loc[max_idx_2000, "DateTime"]
            try:
                default_index_2000 = time_list_2000.index(peak_time_2000)
            except ValueError:
                default_index_2000 = 0

    # セッションステートに該当キーがなければピーク時間で初期化
    if state_key_2000 not in st.session_state or st.session_state[state_key_2000] not in time_labels_2000:
        st.session_state[state_key_2000] = time_labels_2000[min(default_index_2000, len(time_labels_2000)-1)]

    selected_label_2000 = st.selectbox(
        f"時間選択 (2000年 / {map_mode})", 
        time_labels_2000, 
        key=state_key_2000
    )
    selected_time_2000 = time_list_2000[time_labels_2000.index(selected_label_2000)]
    
    row_data_2000 = target_df_2000[target_df_2000["DateTime"] == selected_time_2000].drop(columns=["DateTime"], errors="ignore")
    
    fig_2000 = create_rain_map(row_data_2000, "2000年 東海豪雨", selected_time_2000, map_mode)
    st.pyplot(fig_2000, clear_figure=True)
    plt.close(fig_2000)

# 2026年 9月豪雨エリア
with col2:
    st.markdown(f"####  2026年 9月豪雨")
    time_list_2026 = target_df_2026["DateTime"].tolist()
    time_labels_2026 = [t.strftime('%Y-%m-%d %H:%M') if isinstance(t, pd.Timestamp) else str(t) for t in time_list_2026]
    
    default_index_2026 = 0
    if selected_loc in target_df_2026.columns and not target_df_2026.empty:
        s_vals = pd.to_numeric(target_df_2026[selected_loc], errors='coerce')
        if not s_vals.isna().all():
            max_idx_2026 = s_vals.idxmax()
            peak_time_2026 = target_df_2026.loc[max_idx_2026, "DateTime"]
            try:
                default_index_2026 = time_list_2026.index(peak_time_2026)
            except ValueError:
                default_index_2026 = 0

    # セッションステートに該当キーがなければピーク時間で初期化
    if state_key_2026 not in st.session_state or st.session_state[state_key_2026] not in time_labels_2026:
        st.session_state[state_key_2026] = time_labels_2026[min(default_index_2026, len(time_labels_2026)-1)]

    selected_label_2026 = st.selectbox(
        f"時間選択 (2026年 / {map_mode})", 
        time_labels_2026, 
        key=state_key_2026
    )
    selected_time_2026 = time_list_2026[time_labels_2026.index(selected_label_2026)]
    
    row_data_2026 = target_df_2026[target_df_2026["DateTime"] == selected_time_2026].drop(columns=["DateTime"], errors="ignore")
    
    fig_2026 = create_rain_map(row_data_2026, "2026年 9月豪雨", selected_time_2026, map_mode)
    st.pyplot(fig_2026, clear_figure=True)
    plt.close(fig_2026)
# #おまけ
# #streamlit run main.pyでローカルで確認できる。
# #初期設定①git init, ②git remote add origin https:/// 
# #　①git add . ②git commit -m"◯◯◯" .　③git push origin main
