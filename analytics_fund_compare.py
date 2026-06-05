"""
Fund Compare specific analytics.

Overrides / adds plotting helpers used ONLY by the Fund Compare page.
These plot each selected fund individually against the benchmark
(no category average / median), unlike the shared functions in analytics.py
which the Fund Analysis page relies on.

Import order matters: import this AFTER `from analytics import *` so these
definitions take precedence for the Fund Compare page.
"""

from analytics import *
import pandas as pd
import plotly.express as px


def plot_calendar_returns_multi_comp(fund_df, category_codes, bench_df):
    # ---- Each selected fund individually + benchmark (no average) ----
    cal_list = []
    for code in category_codes:
        df = fetch_nav_data(code)
        cal_list.append(
            calendar_year_returns(df).rename(get_fund_name(int_master_df, code))
        )

    combined = pd.concat(cal_list, axis=1)

    funds_long = combined.reset_index().melt(
        id_vars="year", var_name="Name", value_name="Return"
    )

    bench = calendar_year_returns(bench_df).rename("Benchmark")
    bench_long = bench.reset_index().melt(
        id_vars="year", var_name="Name", value_name="Return"
    )

    plot_df = pd.concat(
        [funds_long, bench_long],
        ignore_index=True
    ).dropna()

    # ---- plot ----
    fig = px.bar(
        plot_df,
        x="year",
        y="Return",
        color="Name",
        barmode="group",
        title="Calendar Year Returns (Selected Funds vs Benchmark)",
        labels={"Return": "Return (%)", "year": "Year"}
    )

    fig.update_layout(height=420)
    return fig


def plot_cagr_summary_multi_comp(fund_df, category_codes, bench_df):
    # ---- Each selected fund individually + benchmark (no average) ----
    data = {}
    for code in category_codes:
        df = fetch_nav_data(code)
        data[get_fund_name(int_master_df, code)] = cagr_summary(df)

    bench = cagr_summary(bench_df)
    period_order = list(bench.index)

    df = pd.DataFrame(data)
    df["Benchmark"] = bench
    df = df.reset_index().rename(columns={"index": "Period"})

    plot_df = df.melt(
        id_vars="Period", var_name="Name", value_name="CAGR"
    ).dropna()

    fig = px.bar(
        plot_df,
        x="Period",
        y="CAGR",
        color="Name",
        barmode="group",
        title="CAGR Comparison (Selected Funds vs Benchmark)",
        labels={"CAGR": "CAGR (%)"},
        category_orders={"Period": period_order}
    )

    fig.update_layout(height=420)
    return fig


def plot_rolling_returns_multi_comp(
    fund_df, category_codes, bench_df, window_years
):
    # ---- Each selected fund individually + benchmark (no average) ----
    fig = px.line(
        title=f"{window_years}Y Rolling Returns (Selected Funds vs Benchmark)"
    )

    for code in category_codes:
        df = fetch_nav_data(code)
        rr = rolling_returns(df, window_years)

        if rr.empty:
            continue

        col = [c for c in rr.columns if "Rolling CAGR" in c][0]
        fig.add_scatter(
            x=rr["start_date"],
            y=rr[col],
            name=get_fund_name(int_master_df, code),
            mode="lines"
        )

    bench_rr = rolling_returns(bench_df, window_years)
    if not bench_rr.empty:
        bcol = [c for c in bench_rr.columns if "Rolling CAGR" in c][0]
        fig.add_scatter(
            x=bench_rr["start_date"],
            y=bench_rr[bcol],
            name="Benchmark",
            mode="lines"
        )

    fig.update_layout(
        height=400,
        xaxis_title="Start Date",
        yaxis_title="CAGR (%)"
    )

    return fig
