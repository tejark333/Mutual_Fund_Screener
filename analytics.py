import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import streamlit as st
from io import StringIO
import zipfile
import io
import requests
import plotly.express as px
import re


plt.rcParams.update({
    "figure.figsize": (9, 4.5),   # ~75% size
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6
})

RATING_ORDER = {
    "Weak": 1,
    "Average": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5,
}

RATING_COLORS = {
    1: "#d73027",   # Weak - red
    2: "#fc8d59",   # Average - orange
    3: "#fee08b",   # Good - yellow
    4: "#91cf60",   # Very Good - light green
    5: "#1a9850",   # Excellent - dark green
}
# -----------------------------
# AMFI MASTER (LIVE)
# -----------------------------

TRI_BENCHMARKS = {
    "Nifty 50 TRI": "Nifty_50_TRI.csv",
    "Nifty 100 TRI": "147518",
    "Nifty 500 TRI": "Nifty_500_TRI.csv",
    "Nifty Midcap 150 TRI": "Nifty_Midcap_150_TRI.csv",
    "Nifty Smallcap 250 TRI": "Nifty_Smallcap_250_TRI.csv",
}


@st.cache_data(ttl=24 * 60 * 60)
def load_amfi_master():


    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    response = requests.get(url)
    response.raise_for_status()

    records = []
    current_category = None

    for line in response.text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Detect category header
        if "Open Ended Schemes" in line:
            # Extract clean category name
            match = re.search(r"\(\s*(.*?)\s*\)", line)
            if match:
                current_category = match.group(1)
            else:
                current_category = "Other"
            continue

        parts = line.split(";")

        # Scheme rows
        if len(parts) >= 4 and parts[0].isdigit():
            records.append({
                "amfi_code": parts[0],
                "scheme_name": parts[3].strip(),
                "category": current_category
            })

    df = pd.DataFrame(records)

    # Drop any malformed rows
    df = df.dropna(subset=["category"])

    return df
int_master_df = load_amfi_master()


def get_categories(master_df):
    return sorted(master_df["category"].dropna().unique())

@st.cache_data
def load_clean_aum():
    return pd.read_csv("data/clean_fund_level_aum_direct_growth.csv")

def get_top_n_by_aum(master_df, aum_df, category, n):
    df = master_df[master_df["category"] == category].copy()

    # ---- FIX: normalize merge key types
    df["amfi_code"] = df["amfi_code"].astype(str)
    aum_df["amfi_code"] = aum_df["amfi_code"].astype(str)

    df = df.merge(
        aum_df,
        on="amfi_code",
        how="inner"
    )

    top_df = df.sort_values("aum_cr", ascending=False).head(n)

    return sorted(top_df["amfi_code"].tolist()), top_df

def add_aum_to_funds(master_df, aum_df, selected_codes):
    df = master_df[master_df["amfi_code"].isin(selected_codes)]

    # ---- FIX: normalize merge key types
    df["amfi_code"] = df["amfi_code"].astype(str)
    aum_df["amfi_code"] = aum_df["amfi_code"].astype(str)

    df = df.merge(
        aum_df,
        on="amfi_code",
        how="inner"
    )

    top_df = df.sort_values("aum_cr", ascending=False)

    return top_df

def consistency_for_top_n(
    top_df,
    category_baseline_df,
    window_years,
    metric="median"   # NEW: "median" or "mean"
):
    results = []

    for _, row in top_df.iterrows():
        amfi_code = str(row["amfi_code"])
        fund_full_name = row["scheme_name"]
        fund_name = extract_fund_house(row["scheme_name"])


        nav_df = fetch_nav_data(amfi_code)
        rr_df = rolling_returns(nav_df, window_years)

        if rr_df.empty or category_baseline_df.empty:
            continue

        col = [c for c in rr_df.columns if "Rolling CAGR" in c][0]

        merged = rr_df.merge(
            category_baseline_df,
            on="start_date",
            how="inner"
        )

        if merged.empty:
            continue

        threshold = merged[metric]

        consistency = (
            (merged[col] >= threshold).sum()
            / len(merged)
            * 100
        )

        results.append({
            "amfi_code": amfi_code,
            "fund_name": fund_name,
            "fund_full_name": fund_full_name,
            "consistency": round(consistency, 2)
        })

    return pd.DataFrame(results)



def category_aum_coverage(aum_df, category_df, top_df):
    total_aum = aum_df[
        aum_df["amfi_code"].isin(category_df["amfi_code"])
    ]["aum_cr"].sum()

    top_aum = top_df["aum_cr"].sum()

    return round((top_aum / total_aum) * 100, 2) if total_aum > 0 else None

def extract_fund_house(fund_name):
    return fund_name.split()[0] + " MF"



def get_fund_types(master_df):
    def classify(cat):
        if "Equity" in cat:
            return "Equity"
        if "Debt" in cat:
            return "Debt"
        if "Hybrid" in cat:
            return "Hybrid"
        if "Index" in cat:
            return "Index"
        return "Other"

    df = master_df.copy()
    df["fund_type"] = df["category"].apply(classify)
    return df


def get_funds_by_category(master_df, category):
    df = master_df[master_df["category"] == category]

    # ✅ Keep only Direct Growth plans
    df = df[
        df["scheme_name"].str.contains("Direct", case=False, na=False) &
        df["scheme_name"].str.contains("Growth", case=False, na=False)
    ]

    return df.sort_values("scheme_name")
    
def get_direct_growth_funds(master_df):
    # ✅ Keep only Direct Growth plans
    df = master_df.copy()
    df = df[
        df["scheme_name"].str.contains("Direct", case=False, na=False) &
        df["scheme_name"].str.contains("Growth", case=False, na=False)
    ]

    return df.sort_values("scheme_name")

def get_amfi_code(master_df, category, fund_name):
    if category == "NA":
        return master_df[
            (master_df["scheme_name"] == fund_name)
    ]["amfi_code"].values[0]
    else:
        return master_df[
            (master_df["category"] == category) &
            (master_df["scheme_name"] == fund_name)
    ]["amfi_code"].values[0]

def get_fund_name(master_df, amfi_code):

    return master_df[
        (master_df["amfi_code"] == amfi_code)
]["scheme_name"].values[0]


# -----------------------------
# NAV DATA
# -----------------------------
@st.cache_data
def fetch_nav_data(amfi_code):
    url = f"https://api.mfapi.in/mf/{amfi_code}"
    data = requests.get(url).json()

    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = df["nav"].astype(float)
    df.sort_values("date", inplace=True)
    return df


# -----------------------------
# ANALYTICS
# -----------------------------
def calendar_year_returns(nav_df):
    nav_df["year"] = nav_df["date"].dt.year
    yearly = nav_df.groupby("year").agg(first=("nav", "first"), last=("nav", "last"))
    yearly["return"] = (yearly["last"] / yearly["first"] - 1) * 100
    return yearly["return"]

def calculate_cagr(df, years):
    end_date = df["date"].max()
    start_date = end_date - pd.DateOffset(years=years)

    period_df = df[df["date"] >= start_date]

    if len(period_df) < 2:
        return None

    start_nav = period_df.iloc[0]["nav"]
    end_nav = df.iloc[-1]["nav"]

    cagr = (end_nav / start_nav) ** (1 / years) - 1
    return float(round(cagr * 100, 2))


def cagr_summary(df):
    periods = [1, 3, 5, 7, 10]
    return pd.Series(
        {f"{p}Y CAGR (%)": calculate_cagr(df, p) for p in periods}
    )


def rolling_returns(df, window_years):
    # temp = df.copy()
    # temp.set_index("date", inplace=True)

    temp = (
    df.sort_values("date")
      .set_index("date")
      .resample("M")
      .last()
      .dropna()
    )


    last_date = temp.index[-1]
    results = []

    for start_date in temp.index:
        target_date = start_date + pd.DateOffset(years=window_years)

        # Stop if full window not available
        if target_date > last_date:
            break

        idx = temp.index.searchsorted(target_date, side="right") - 1

        if idx <= temp.index.get_loc(start_date):
            continue

        start_nav = temp.loc[start_date, "nav"]
        end_nav = temp.iloc[idx]["nav"]

        cagr = (end_nav / start_nav) ** (1 / window_years) - 1

        results.append({
            "start_date": start_date,
            "end_date": temp.index[idx],
            f"{window_years}Y Rolling CAGR (%)": round(cagr * 100, 2)
        })

    return pd.DataFrame(results)


def filter_to_last_full_month(nav_df, end_date):
    nav_df = nav_df[nav_df["date"] <= end_date].copy()
    return nav_df

def monthly_returns_old(nav_df):
    df = nav_df.copy()
    df.set_index("date", inplace=True)

    monthly = df["nav"].resample("M").last().pct_change() * 100
    return monthly.dropna()

def monthly_returns(nav_df):
    df = nav_df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    monthly_nav = (
        df.sort_values("date")
          .groupby("year_month")["nav"]
          .last()
    )

    monthly_ret = monthly_nav.pct_change()
    return monthly_ret.dropna() * 100

def read_tri(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df.Date,format="%d %b %Y")
    # df['Total Returns Index'] = df.Price.str.replace(",","")
    df['Total Returns Index'] = df['Total Returns Index'].apply(lambda x:float(x))
    df = df.sort_values("Date")
    df.rename({'Date':'date','Total Returns Index':'nav'},inplace=True,axis=1)
    print(df.info())
    return df

def risk_metrics_monthly(
    nav_df,
    bench_df,
    end_date,
    lookback_months=36,
    risk_free_rate_annual=0.06  # 6% default (can tweak later)
):
    # ---- Cut to last full month
    nav_df = nav_df[nav_df["date"] <= end_date].copy()
    bench_df = bench_df[bench_df["date"] <= end_date].copy()

    # ---- Monthly returns (%)
    fund_m = monthly_returns(nav_df).tail(lookback_months)
    bench_m = monthly_returns(bench_df).tail(lookback_months)

    df = pd.concat([fund_m, bench_m], axis=1).dropna()
    df.columns = ["fund_pct", "bench_pct"]

    if len(df) < lookback_months:
        return None
    # -------------------------------------------------
    # UPSIDE & DOWNSIDE CAPTURE RATIOS
    # -------------------------------------------------
    
    # Upside capture: benchmark > 0
    up_mask = df["bench_pct"] > 0
    if up_mask.any():
        upside_capture = (
            df.loc[up_mask, "fund_pct"].mean()
            / df.loc[up_mask, "bench_pct"].mean()
        ) * 100
    else:
        upside_capture = None
    
    # Downside capture: benchmark < 0
    down_mask = df["bench_pct"] < 0
    if down_mask.any():
        downside_capture = (
            df.loc[down_mask, "fund_pct"].mean()
            / df.loc[down_mask, "bench_pct"].mean()
        ) * 100
    else:
        downside_capture = None


    # ---- Convert % → decimal (CRITICAL FIX)
    fund = df["fund_pct"] / 100.0
    bench = df["bench_pct"] / 100.0

    # ---- Risk-free rate (monthly, decimal)
    rf_monthly = (1 + risk_free_rate_annual) ** (1 / 12) - 1

    # ---- Annualised mean return
    mean_ret = fund.mean() * 12

    # ---- Annualised volatility
    std_dev = fund.std() * np.sqrt(12)

    # ---- Downside deviation
    downside = fund[fund < rf_monthly]
    downside_std = downside.std() * np.sqrt(12) if not downside.empty else None

    # ---- Sharpe & Sortino (FIXED)
    sharpe = (mean_ret - risk_free_rate_annual) / std_dev if std_dev else None
    sortino = (
        (mean_ret - risk_free_rate_annual) / downside_std
        if downside_std else None
    )

    # -------------------------------------------------
    # BETA & ALPHA (ValueResearch definition)
    # -------------------------------------------------
    
    # Monthly decimal returns (CRITICAL)
    fund_dec = df["fund_pct"].values / 100.0
    bench_dec = df["bench_pct"].values / 100.0
    
    # Defensive checks
    if np.var(bench_dec) == 0 or len(fund_dec) < 12:
        beta = None
        alpha = None
    else:
        # Beta = Cov(fund, benchmark) / Var(benchmark)
        covariance = np.cov(fund_dec, bench_dec, ddof=1)[0, 1]
        variance = np.var(bench_dec, ddof=1)
    
        beta = covariance / variance
    
        # Alpha (monthly, then annualised)
        alpha_monthly = np.mean(fund_dec) - beta * np.mean(bench_dec)
        alpha = alpha_monthly * 12

    return {
        "Mean Return (%)": round(mean_ret * 100, 2),
        "Std Dev (%)": round(std_dev * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Beta": round(beta, 2),
        "Alpha (%)": round(alpha * 100, 2),
        "Upside Capture (%)": round(upside_capture, 2),
        "Downside Capture (%)": round(downside_capture, 2)
    }


def category_risk_metrics_monthly(
    category_amfi_codes,
    bench_df,
    end_date,
    lookback_months=36
):
    metrics = []
    codes = []
    for code in category_amfi_codes:
        nav_df = fetch_nav_data(code)
        m = risk_metrics_monthly(
            nav_df,
            bench_df,
            end_date,
            lookback_months
        )
        if m:
            metrics.append(m)
            codes.append(code)
        else:
            print(code)

    if not metrics:
        return None
        
    return (pd.DataFrame(metrics,index=codes).reset_index().rename({"index":"amfi_code",
                                                                                  "Sharpe": "sharpe",
                                                                                  "Sortino": "sortino",
                                                                                  "Alpha (%)": "alpha",
                                                                                  "Upside Capture (%)": "upside_capture",
                                                                                  "Downside Capture (%)": "downside_capture"},axis=1),
        pd.DataFrame(metrics)
        .mean()
        .round(2)
        .to_dict()
    )


def best_worst_rolling_periods(rr_df, col):
    return {
        "best_period": rr_df.loc[rr_df[col].idxmax()],
        "worst_period": rr_df.loc[rr_df[col].idxmin()]
    }


def rolling_return_percentiles(rr_df, col):
    return {
        "10%": round(rr_df[col].quantile(0.1), 2),
        "25%": round(rr_df[col].quantile(0.25), 2),
        "50%": round(rr_df[col].quantile(0.5), 2),
        "75%": round(rr_df[col].quantile(0.75), 2),
        "90%": round(rr_df[col].quantile(0.9), 2)
    }

@st.cache_data(ttl=24 * 60 * 60)
def category_rolling_baseline(category_amfis, window_years):
    rolling_list = []

    for category_amfi in category_amfis:
        df = fetch_nav_data(category_amfi)

        if (df["date"].max() - df["date"].min()).days < window_years * 365:
            continue
        
        rr = rolling_returns(df, window_years)

        # ---- SAFETY CHECK
        if rr is None or rr.empty:
            continue

        cagr_cols = [c for c in rr.columns if "Rolling CAGR" in c]
        if not cagr_cols:
            continue

        cagr_col = cagr_cols[0]

        rolling_list.append(
            rr[["start_date", cagr_col]].rename(columns={cagr_col: "cagr"})
        )

    # ---- FINAL SAFETY
    if not rolling_list:
        return pd.DataFrame(columns=["start_date", "mean", "median"])

    combined = (
        pd.concat(rolling_list)
        .groupby("start_date")["cagr"]
        .agg(["mean", "median"])
        .reset_index()
    )

    return combined


def consistency_vs_category(fund_rr_df, category_baseline_df, window_years):
    col = f"{window_years}Y Rolling CAGR (%)"

    merged = fund_rr_df.merge(
        category_baseline_df,
        on="start_date",
        how="inner"
    )

    total = len(merged)

    median_score = (
        (merged[col] >= merged["median"]).sum() / total * 100
        if total > 0 else None
    )

    mean_score = (
        (merged[col] >= merged["mean"]).sum() / total * 100
        if total > 0 else None
    )

    return {
        "median_consistency": round(median_score, 2),
        "mean_consistency": round(mean_score, 2)
    }


# -----------------------------
# Category
# -----------------------------

def category_calendar_year_returns(category_amfi_codes):
    cal_list = []

    for code in category_amfi_codes:
        df = fetch_nav_data(code)
        cal = calendar_year_returns(df)
        cal_list.append(cal.rename(code))

    combined = pd.concat(cal_list, axis=1)
    return combined.median(axis=1),combined

def category_cagr_summary(category_amfi_codes):
    periods = [1, 3, 5, 7, 10]
    out = {}
    per_fund = {}

    for p in periods:
        vals = []
        for code in category_amfi_codes:
            df = fetch_nav_data(code)
            v = calculate_cagr(df, p)
            if v is not None:
                vals.append(v)

        out[f"{p}Y CAGR (%)"] = np.median(vals) if vals else None

    return pd.Series(out)

def category_rolling_returns(category_amfi_codes, window_years):
    rr_list = []

    for code in category_amfi_codes:
        df = fetch_nav_data(code)
        rr = rolling_returns(df, window_years)

        if rr.empty:
            continue

        col = [c for c in rr.columns if "Rolling CAGR" in c][0]
        rr_list.append(rr.set_index("start_date")[col])

    combined = pd.concat(rr_list, axis=1)
    out = combined.median(axis=1).reset_index()
    out.columns = ["start_date", col]

    return out


# Composite score

def rank_to_score(series, higher_is_better=True):
    """
    Converts a metric series into 0–100 percentile score
    """
    s = series.copy()

    # Handle direction
    if higher_is_better:
        ranks = s.rank(ascending=False, method="average")
    else:
        ranks = s.rank(ascending=True, method="average")

    n = len(s)
    scores = 100 * (n - ranks) / (n - 1)
    return scores

def score_label(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Very Good"
    elif score >= 55:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Weak"


def compute_composite_score(category_df):
    df = category_df.copy()

    # -----------------------------
    # Percentile scores
    # -----------------------------
    df["consistency_score"] = rank_to_score(
        df["consistency_pct"], higher_is_better=True
    )

    df["downside_capture_score"] = rank_to_score(
        df["downside_capture"], higher_is_better=False   # LOWER is better
    )

    df["upside_capture_score"] = rank_to_score(
        df["upside_capture"], higher_is_better=True
    )

    df["sortino_score"] = rank_to_score(
        df["sortino"], higher_is_better=True
    )

    df["sharpe_score"] = rank_to_score(
        df["sharpe"], higher_is_better=True
    )

    df["alpha_score"] = rank_to_score(
        df["alpha"], higher_is_better=True
    )

    # -----------------------------
    # Weighted composite score
    # -----------------------------
    df["composite_score"] = (
        0.30 * df["consistency_score"] +
        0.20 * df["downside_capture_score"] +
        0.15 * df["upside_capture_score"] +
        0.15 * df["sortino_score"] +
        0.10 * df["sharpe_score"] +
        0.10 * df["alpha_score"]
    )

    df["rating"] = df["composite_score"].apply(score_label)

    return df.sort_values("composite_score", ascending=False)

def compute_rating_snapshot(
    cutoff_date,
    nav_df,
    bench_df,
    top_amfi_codes,
    cons_df,
    master_df,
):
    ind_cat_metrics, _ = category_risk_metrics_monthly(
        top_amfi_codes, bench_df, cutoff_date
    )

    ind_cat_metrics["amfi_code"] = ind_cat_metrics["amfi_code"].astype(str)

    table_df = cons_df.sort_values("aum_cr", ascending=False)
    table_df["aum_rank"] = range(1, len(table_df) + 1)

    ind_cat_metrics = (
        ind_cat_metrics
        .merge(
            table_df[["amfi_code", "aum_cr", "aum_rank", "consistency"]],
            how="left",
            on="amfi_code"
        )
        .merge(
            master_df[["amfi_code", "scheme_name"]],
            how="left",
            on="amfi_code"
        )
        .rename({"consistency": "consistency_pct"}, axis=1)
    )

    scored = compute_composite_score(ind_cat_metrics)

    scored["rating"] = scored["composite_score"].apply(score_label)
    scored["rating_level"] = scored["rating"].map(RATING_ORDER)
    scored["period_end"] = cutoff_date

    return scored[
        ["scheme_name", "period_end", "composite_score", "rating", "rating_level"]
    ]

def get_month_end_dates(end_date, n=6):
    end = pd.to_datetime(end_date)
    return sorted(
        (end - pd.DateOffset(months=i)).to_period("M").to_timestamp("M")
        for i in range(n)
    )


def get_quarter_end_dates(end_date, n=8):
    end = pd.to_datetime(end_date)
    return sorted(
        (end - pd.DateOffset(months=3 * i)).to_period("Q").to_timestamp("Q")
        for i in range(n)
    )

def get_year_end_dates(end_date, n=5):
    end = pd.to_datetime(end_date)
    return sorted(
        (end - pd.DateOffset(months=12 * i)).to_period("Y").to_timestamp("Y")
        for i in range(n)
    )


def build_monthly_rating_history(
    nav_df,
    bench_df,
    top_amfi_codes,
    cons_df,
    master_df,
    end_date,
    months=6,
):
    dates = get_month_end_dates(end_date, months)

    return pd.concat(
        [
            compute_rating_snapshot(
                d, nav_df, bench_df, top_amfi_codes, cons_df, master_df
            )
            for d in dates
        ],
        ignore_index=True,
    )

def build_quarterly_rating_history(
    nav_df,
    bench_df,
    top_amfi_codes,
    cons_df,
    master_df,
    end_date,
    quarters=8,
):
    dates = get_quarter_end_dates(end_date, quarters)

    return pd.concat(
        [
            compute_rating_snapshot(
                d, nav_df, bench_df, top_amfi_codes, cons_df, master_df
            )
            for d in dates
        ],
        ignore_index=True,
    )

def build_yearly_rating_history(
    nav_df,
    bench_df,
    top_amfi_codes,
    cons_df,
    master_df,
    end_date,
    years=5,
):
    dates = get_year_end_dates(end_date, years)

    return pd.concat(
        [
            compute_rating_snapshot(
                d, nav_df, bench_df, top_amfi_codes, cons_df, master_df
            )
            for d in dates
        ],
        ignore_index=True,
    )



# -----------------------------
# PLOTS (FIGURE-BASED)
# -----------------------------
def plot_calendar_year_returns(cal_df):
    df = cal_df.reset_index()
    df.columns = ["Year", "Return"]

    fig = px.bar(
        df,
        x="Year",
        y="Return",
        text="Return",
        labels={"Return": "Return (%)"}
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=500)

    return fig

def plot_calendar_returns_multi(fund_df, category_codes, bench_df):
    fund = calendar_year_returns(fund_df)
    cat,cat_indi = category_calendar_year_returns(category_codes)
    bench = calendar_year_returns(bench_df)

    df = pd.DataFrame({
        "Fund": fund,
        "Category": cat,
        "Benchmark": bench
    }).dropna().reset_index()

    fig = px.bar(
        df,
        x="year",
        y=["Fund", "Category", "Benchmark"],
        barmode="group",
        title="Calendar Year Returns",
        labels={"value": "Return (%)", "year": "Year"}
    )

    fig.update_layout(height=380)
    return fig

def plot_calendar_returns_multi_comp(fund_df, category_codes, bench_df):
    # Individual returns
    fund = calendar_year_returns(fund_df).rename("Selected Fund")
    bench = calendar_year_returns(bench_df).rename("Benchmark")
    cat,cat_indi = category_calendar_year_returns(category_codes)

    # ---- reshape to long format ----
    fund_df_long = fund.reset_index().melt(
        id_vars="year", var_name="Name", value_name="Return"
    )
    fund_df_long["Type"] = "Fund"

    bench_df_long = bench.reset_index().melt(
        id_vars="year", var_name="Name", value_name="Return"
    )
    bench_df_long["Type"] = "Benchmark"

    cat_df_long = cat_indi.reset_index().melt(
        id_vars="year", var_name="Name", value_name="Return"
    )
    cat_df_long['Name'] = cat_df_long['Name'].apply(lambda x:get_fund_name(int_master_df,x))
    cat_df_long["Type"] = "Category"

    # Combine everything
    plot_df = pd.concat(
        [fund_df_long, cat_df_long, bench_df_long],
        ignore_index=True
    ).dropna()

    # ---- plot ----
    fig = px.bar(
        plot_df,
        x="year",
        y="Return",
        color="Name",
        barmode="group",
        title="Calendar Year Returns (Fund vs Category Funds vs Benchmark)",
        labels={"Return": "Return (%)", "year": "Year"}
    )

    fig.update_layout(height=420)
    return fig




def plot_cagr_summary(cagr_series):
    df = cagr_series.dropna().reset_index()
    df.columns = ["Period", "CAGR"]

    fig = px.bar(
        df,
        x="Period",
        y="CAGR",
        text="CAGR",
        labels={"CAGR": "CAGR (%)"}
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(height=500)

    return fig

def plot_cagr_summary_multi(fund_df, category_codes, bench_df):
    fund = cagr_summary(fund_df)
    cat = category_cagr_summary(category_codes)
    bench = cagr_summary(bench_df)

    df = pd.DataFrame({
        "Period": fund.index,
        "Fund": fund.values,
        "Category": cat.values,
        "Benchmark": bench.values
    })

    fig = px.bar(
        df,
        x="Period",
        y=["Fund", "Category", "Benchmark"],
        barmode="group",
        title="CAGR Comparison",
        labels={"value": "CAGR (%)"}
    )

    fig.update_layout(height=380)
    return fig



def plot_rolling_returns(rr_df, col, window):
    fig = px.line(
        rr_df,
        x="start_date",
        y=col,
        labels={
            "start_date": "Start Date",
            col: "CAGR (%)"
        }
    )

    fig.update_layout(height=500)

    return fig

def plot_rolling_returns_multi(
    fund_df, category_codes, bench_df, window_years
):
    fund_rr = rolling_returns(fund_df, window_years)
    cat_rr = category_rolling_returns(category_codes, window_years)
    bench_rr = rolling_returns(bench_df, window_years)

    col = [c for c in fund_rr.columns if "Rolling CAGR" in c][0]

    fig = px.line(title=f"{window_years}Y Rolling Returns")

    fig.add_scatter(
        x=fund_rr["start_date"], y=fund_rr[col], name="Fund"
    )
    fig.add_scatter(
        x=cat_rr["start_date"], y=cat_rr[col], name="Category"
    )
    fig.add_scatter(
        x=bench_rr["start_date"], y=bench_rr[col], name="Benchmark"
    )

    fig.update_layout(
        height=400,
        xaxis_title="Start Date",
        yaxis_title="CAGR (%)"
    )

    return fig


def plot_rolling_return_distribution(rr_df, col, percentiles):
    fig = px.histogram(
        rr_df,
        x=col,
        nbins=30,
        title="Rolling Return Distribution",
        labels={col: "CAGR (%)"}
    )

    for p, v in percentiles.items():
        fig.add_vline(
            x=v,
            line_dash="dash",
            annotation_text=p,
            annotation_position="top"
        )

    fig.update_layout(height=350)

    return fig

def plot_consistency_vs_aum_rank(cons_df, selected_amfi_code):
    cons_df = cons_df.sort_values("aum_cr", ascending=False)
    cons_df["aum_rank"] = range(1, len(cons_df) + 1)

    cons_df["is_selected"] = cons_df["amfi_code"] == selected_amfi_code

    fig = px.scatter(
        cons_df,
        x="aum_rank",
        y="consistency",
        text="fund_name",
        color="is_selected",
        color_discrete_map={True: "red", False: "blue"},
        labels={
            "aum_rank": "AUM Rank (1 = Largest)",
            "consistency": "Consistency (%)"
        },
        title="Consistency vs AUM Rank (3Y)",
        hover_data={
            "fund_name": True,
            "aum_cr": True,
            "consistency": True,
            "aum_rank": True
        }
    )

    fig.update_traces(
        textposition="top center",
        textfont_size=10
    )

    fig.update_layout(
        showlegend=False,
        height=450
    )

    return fig

def plot_rating_signal(df, title):
    fig = px.line(
        df.sort_values("period_end"),
        x="period_end",
        y="rating_level",
        color="scheme_name",
        line_shape="hv",
        markers=True,
        title=title,
    )

    fig.update_layout(
        height=550,
        yaxis=dict(
            tickmode="array",
            tickvals=list(RATING_ORDER.values()),
            ticktext=list(RATING_ORDER.keys()),
            title="Rating",
        ),
        xaxis_title="Period End",

        # ---- LEGEND SETTINGS ----
        legend=dict(
            title="Funds",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,          # move legend outside plot
            itemsizing="constant",
            font=dict(size=11),
        ),
        margin=dict(r=300),  # space for legend
    )

    return fig

def plot_rating_signal_small_multiples(df, title):
    df = df.sort_values(["scheme_name", "period_end"])

    fig = px.line(
        df,
        x="period_end",
        y="rating_level",
        facet_row="scheme_name",
        line_shape="hv",
        markers=True,
        height=120 * df["scheme_name"].nunique(),
        title=title,
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(RATING_ORDER.values()),
        ticktext=list(RATING_ORDER.keys()),
        title=None,
    )

    fig.update_xaxes(title="Period End")

    fig.update_layout(
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    # Clean facet titles: remove "scheme_name="
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])
    )

    return fig

def plot_rating_heatmap(df, title):
    # Pivot: funds x time
    heat_df = df.pivot_table(
        index="scheme_name",
        columns="period_end",
        values="rating_level",
        aggfunc="last"
    )

    # Ensure chronological order (columns)
    heat_df = heat_df.sort_index(axis=1)

    # ---- SORT ROWS: highest latest rating on top ----
    latest_col = heat_df.columns[-1]
    heat_df = heat_df.sort_values(
        by=latest_col,
        ascending=False
    )

    # Discrete colorscale
    colorscale = [
        [0.00, "#d73027"], [0.20, "#d73027"],  # Weak
        [0.20, "#fc8d59"], [0.40, "#fc8d59"],  # Average
        [0.40, "#fee08b"], [0.60, "#fee08b"],  # Good
        [0.60, "#91cf60"], [0.80, "#91cf60"],  # Very Good
        [0.80, "#1a9850"], [1.00, "#1a9850"],  # Excellent
    ]

    fig = px.imshow(
        heat_df,
        aspect="auto",
        color_continuous_scale=colorscale,
        zmin=1,
        zmax=5,
        title=title,
    )

    fig.update_layout(
        height=30 * len(heat_df) + 200,
        coloraxis_colorbar=dict(
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["Weak", "Average", "Good", "Very Good", "Excellent"],
            title="Rating",
        ),
        xaxis_title="Date",
        yaxis_title="Fund",
    )

    return fig
