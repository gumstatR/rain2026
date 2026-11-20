import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="住宅ローン返済シミュレーター", layout="wide")
st.title("🏠 住宅ローン返済シミュレーターWebアプリ")
st.caption("サイドバーで借入条件を変更すると、返済額と推移グラフがリアルタイムに更新されます。")

# --- サイドバー：借入条件の入力 ---
st.sidebar.header("借入条件の設定")

loan_amount_man = st.sidebar.number_input(
    "借入額（万円）",
    min_value=100,
    max_value=30000,
    value=4000,
    step=100,
)
annual_rate = st.sidebar.number_input(
    "年利（%）",
    min_value=0.0,
    max_value=15.0,
    value=1.2,
    step=0.1,
    format="%.2f",
)
years = st.sidebar.slider(
    "返済期間（年）",
    min_value=1,
    max_value=50,
    value=35,
    step=1,
)
repayment_type = st.sidebar.radio(
    "返済方式",
    options=["元利均等返済", "元金均等返済"],
)
chart_unit = st.sidebar.radio(
    "グラフの単位",
    options=["年単位（年間の合計支払額）", "月単位（1〜120ヶ月目など抜粋表示）"],
)

# --- 計算処理 ---
principal = loan_amount_man * 10_000  # 円換算
months = years * 12
monthly_rate = (annual_rate / 100) / 12

schedule = []
balance = float(principal)

if repayment_type == "元利均等返済":
    if monthly_rate == 0:
        monthly_payment = principal / months
    else:
        factor = (1 + monthly_rate) ** months
        monthly_payment = principal * (monthly_rate * factor) / (factor - 1)

    for m in range(1, months + 1):
        interest = balance * monthly_rate
        principal_paid = monthly_payment - interest
        # 最終月の端数調整
        if m == months or principal_paid > balance:
            principal_paid = balance
            payment = principal_paid + interest
        else:
            payment = monthly_payment
        balance = max(0.0, balance - principal_paid)
        year_num = (m - 1) // 12 + 1
        schedule.append({
            "month": m,
            "year": year_num,
            "payment": payment,
            "principal": principal_paid,
            "interest": interest,
            "balance": balance,
        })

    first_month_payment = schedule[0]["payment"]
    last_month_payment = schedule[-1]["payment"]

else:  # 元金均等返済
    fixed_principal = principal / months
    for m in range(1, months + 1):
        interest = balance * monthly_rate
        principal_paid = balance if m == months else fixed_principal
        payment = principal_paid + interest
        balance = max(0.0, balance - principal_paid)
        year_num = (m - 1) // 12 + 1
        schedule.append({
            "month": m,
            "year": year_num,
            "payment": payment,
            "principal": principal_paid,
            "interest": interest,
            "balance": balance,
        })

    first_month_payment = schedule[0]["payment"]
    last_month_payment = schedule[-1]["payment"]

df = pd.DataFrame(schedule)
total_payment = df["payment"].sum()
total_interest = df["interest"].sum()

# --- メイン画面：サマリー指標 ---
col1, col2, col3 = st.columns(3)

if repayment_type == "元利均等返済":
    col1.metric("毎月の返済額（目安）", f"{first_month_payment:,.0f} 円")
else:
    col1.metric(
        "毎月の返済額（初月 → 最終月）",
        f"{first_month_payment:,.0f} 円",
        delta=f"最終月: {last_month_payment:,.0f} 円",
        delta_color="off",
    )

col2.metric("総支払額", f"{total_payment / 10_000:,.1f} 万円", help=f"{total_payment:,.0f} 円")
col3.metric("利息合計", f"{total_interest / 10_000:,.1f} 万円", help=f"{total_interest:,.0f} 円")

st.divider()

import altair as alt

# --- 棒グラフ表示 ---
st.subheader("📊 返済額の推移（元金・利息の内訳）")

if chart_unit == "年単位（年間の合計支払額）":
    yearly_df = df.groupby("year", as_index=False)[["principal", "interest"]].sum()
    yearly_df["元金（万円）"] = yearly_df["principal"] / 10_000
    yearly_df["利息（万円）"] = yearly_df["interest"] / 10_000

    plot_df = yearly_df.melt(
        id_vars="year",
        value_vars=["元金（万円）", "利息（万円）"],
        var_name="内訳",
        value_name="金額（万円）",
    )
    plot_df["order"] = plot_df["内訳"].map({"元金（万円）": 0, "利息（万円）": 1})

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="経過年数（年）", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("金額（万円）:Q", title="支払額（万円）"),
            color=alt.Color("内訳:N", sort=["元金（万円）", "利息（万円）"]),
            order=alt.Order("order:Q"),
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

else:
    max_m = st.slider("表示する経過月数（1ヶ月目〜）", min_value=12, max_value=months, value=min(60, months), step=12)
    sub_df = df[df["month"] <= max_m].copy()
    sub_df["元金（円）"] = sub_df["principal"]
    sub_df["利息（円）"] = sub_df["interest"]

    plot_df = sub_df.melt(
        id_vars="month",
        value_vars=["元金（円）", "利息（円）"],
        var_name="内訳",
        value_name="金額（円）",
    )
    plot_df["order"] = plot_df["内訳"].map({"元金（円）": 0, "利息（円）": 1})

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("month:O", title="経過月数（ヶ月）", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("金額（円）:Q", title="支払額（円）"),
            color=alt.Color("内訳:N", sort=["元金（円）", "利息（円）"]),
            order=alt.Order("order:Q"),
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)




# --- 明細テーブル（折りたたみ） ---
with st.expander("年次推移データを見る"):
    summary_table = (
        df.groupby("year", as_index=False)
        .agg(
            年間返済額_万円=("payment", lambda s: round(s.sum() / 10_000, 1)),
            元金返済_万円=("principal", lambda s: round(s.sum() / 10_000, 1)),
            利息支払_万円=("interest", lambda s: round(s.sum() / 10_000, 1)),
            年末残高_万円=("balance", lambda s: round(s.iloc[-1] / 10_000, 1)),
        )
        .rename(columns={"year": "経過年数"})
    )
    st.dataframe(summary_table, use_container_width=True, hide_index=True)


    # #streamlit run main.pyでローカルで確認できる。
# #初期設定①git init, ②git remote add ofigin https:/// 
# #　①git add . ②git commit -m"◯◯◯".　③git push origin main
