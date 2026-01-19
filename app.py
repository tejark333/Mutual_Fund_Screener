import streamlit as st
from analytics import *

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    layout="wide",
    page_title="Mutual Fund Analytics"
)

st.title("📊 Mutual Fund Rolling Return Analytics")

# -----------------------------
# LOAD MASTER DATA
# -----------------------------
master_df = load_amfi_master()

# -----------------------------
# INPUTS (MAIN PAGE)
# -----------------------------

st.subheader("Select Mutual Fund")

typed_df = get_fund_types(master_df)

# ---- Row 1: Fund Type + Category
r1c1, r1c2, r1c3 = st.columns([3, 6, 3])

with r1c1:
    fund_type = st.selectbox(
        "Fund Type",
        sorted(typed_df["fund_type"].unique())
    )

with r1c2:
    filtered_df = typed_df[typed_df["fund_type"] == fund_type]
    category = st.selectbox(
        "Mutual Fund Category",
        sorted(filtered_df["category"].unique())
    )

with r1c3:
    benchmark_name = st.selectbox(
    "Benchmark (TRI)",
    list(TRI_BENCHMARKS.keys())
    )




# ---- Row 2: Fund + Rolling window + Button
r2c1, r2c2, r2c3, r2c4 = st.columns([6, 2, 3, 2])

with r2c1:
    fund_df = get_funds_by_category(filtered_df, category)
    fund_name = st.selectbox(
        "Mutual Fund",
        fund_df["scheme_name"].tolist()
    )

with r2c2:
    rolling_window = st.selectbox(
        "Rolling Window (Years)",
        [3, 5]
    )

    
with r2c3:
    top_n = st.selectbox(
        "Top N funds to compare (by AUM)",
        [5, 10, 20],
        index=0
    )


with r2c4:
    st.markdown("###")
    run = st.button("Run Analysis", type="primary")


# -----------------------------
# MAIN ANALYTICS
# -----------------------------

benchmark_code = TRI_BENCHMARKS[benchmark_name]
bench_df = read_tri(R"data\\"+benchmark_code)


aum_df = load_clean_aum()

top_amfi_codes, top_df = get_top_n_by_aum(
        master_df,
        aum_df,
        category,
        top_n
    )

if run:
    amfi_code = get_amfi_code(master_df, category, fund_name)
    nav_df = fetch_nav_data(amfi_code)

    END_DATE = pd.Timestamp("2025-12-31")

    category_funds = master_df[master_df["category"] == category]["amfi_code"]

    category_amfis = [code
        for code in category_funds
    ]

    # ---- Fund Overview
    st.subheader("Fund Overview")
    o1, o2 = st.columns(2)
    o1.metric("Inception Date", str(nav_df["date"].min().date()))
    o2.metric("Latest NAV Date", str(nav_df["date"].max().date()))


    # 3Y rolling returns
    fund_rr = rolling_returns(nav_df, 3)
    bench_rr = rolling_returns(bench_df, 3)

    fund_metrics = risk_metrics_monthly(
        nav_df,
        bench_df,
        END_DATE
    )
    
    bench_metrics = risk_metrics_monthly(
        bench_df,
        bench_df,
        END_DATE
    )
    
    ind_cat_metrics,category_metrics = category_risk_metrics_monthly(
        top_amfi_codes,
        bench_df,
        END_DATE
    )
    

    risk_df = pd.DataFrame.from_dict(
        {
            fund_name: fund_metrics,
            "Benchmark": bench_metrics,
            "Category": category_metrics
        },
        orient="index"
    )


    
    



    # ---- Calendar Returns
    st.subheader("Calendar Year Returns")
    cal_df = calendar_year_returns(nav_df)
    # st.pyplot(plot_calendar_year_returns(cal_df),use_container_width=False)

    c_plot, _ = st.columns([3, 1])  # 75% width
    with c_plot:
        st.plotly_chart(
    plot_calendar_returns_multi(nav_df, top_amfi_codes, bench_df),
    use_container_width=True)



    # ---- CAGR
    st.subheader("CAGR Summary")

    cagr_series = cagr_summary(nav_df)
    # st.write(type(cagr_series))
    c_plot, _ = st.columns([3, 1])  # 75% width
    with c_plot:
        st.plotly_chart(
    plot_cagr_summary_multi(nav_df, top_amfi_codes, bench_df),
    use_container_width=True
)




    # ---- Rolling Returns
    st.subheader(f"{rolling_window}Y Rolling Returns")
    rr_df = rolling_returns(nav_df, rolling_window)
    col_name = f"{rolling_window}Y Rolling CAGR (%)"
    c_plot, _ = st.columns([3, 1])  # 75% width
    with c_plot:
        st.plotly_chart(
    plot_rolling_returns_multi(
        nav_df, top_amfi_codes, bench_df, rolling_window
    ),
    use_container_width=True
)


    # ---- Best & Worst
    st.subheader("Best & Worst Rolling Periods")
    bw = best_worst_rolling_periods(rr_df, col_name)
    st.write("Best Period")
    st.dataframe(bw["best_period"].to_frame())
    st.write("Worst Period")
    st.dataframe(bw["worst_period"].to_frame())

    # ---- Distribution
    st.subheader("Rolling Return Distribution")
    percentiles = rolling_return_percentiles(rr_df, col_name)
    st.json(percentiles)
    c_plot, _ = st.columns([3, 1])  # 75% width
    with c_plot:
        st.plotly_chart(
    plot_rolling_return_distribution(rr_df, col_name, percentiles),
    use_container_width=True
)


    # ---- Consistency across category

    

    fund_aum = aum_df[aum_df["amfi_code"] == amfi_code]["aum_cr"]

    if not fund_aum.empty:
        st.metric("Fund AUM (₹ Cr)", fund_aum.values[0])


    category_df = master_df[master_df["category"] == category]

    

    coverage = category_aum_coverage(aum_df, category_df, top_df)

    if coverage is not None:
        st.metric(f"Top {top_n} Category AUM Coverage (%)", coverage)



    category_amfi_codes = sorted(top_amfi_codes)

    
    for w in [3]:
        fund_rr = rolling_returns(nav_df, w)
        cat_baseline = category_rolling_baseline(category_amfi_codes, w)
    
        scores = consistency_vs_category(fund_rr, cat_baseline, w)
    
        st.subheader(f"{w}Y Consistency vs Category")
        c1, c2 = st.columns(2)
        c1.metric("vs Category Median (%)", scores["median_consistency"])
        c2.metric("vs Category Mean (%)", scores["mean_consistency"])

    consistency_metric = st.radio(
    "Consistency Comparison Basis",
    ["median", "mean"],
    horizontal=True
)

    
    cons_df = consistency_for_top_n(
    top_df,
    cat_baseline,
    3,
    metric=consistency_metric
)

    cons_df = cons_df.merge(
        top_df[["amfi_code", "aum_cr", "scheme_name"]],
        on="amfi_code",
        how="left"
    )

    
    st.subheader("Consistency vs AUM Rank (3Y)")

    c_plot, _ = st.columns([3, 1])
    with c_plot:
        st.plotly_chart(
    plot_consistency_vs_aum_rank(cons_df, amfi_code),
    use_container_width=True
)


    st.subheader("Top Funds – AUM & Consistency")

    table_df = cons_df.sort_values("aum_cr", ascending=False)
    table_df["aum_rank"] = range(1, len(table_df) + 1)
    
    # st.dataframe(
    #     table_df[["amfi_code",
    #         "fund_full_name",
    #         "aum_rank",
    #         "aum_cr",
    #         "consistency"
    #     ]].rename(columns={
    #         "fund_name": "Fund Name",
    #         "aum_rank": "AUM Rank",
    #         "aum_cr": "AUM (₹ Cr)",
    #         "consistency": "Consistency (%)"
    #     })
    # )

    st.subheader("Risk Metrics (Monthly, Last 3 Years)")
    st.dataframe(risk_df)
    
    ind_cat_metrics['amfi_code'] = ind_cat_metrics['amfi_code'].astype('str')
    ind_cat_metrics = ind_cat_metrics.merge(table_df[['amfi_code', 'consistency','aum_rank','aum_cr']],how='left', on='amfi_code')
    ind_cat_metrics = ind_cat_metrics.merge(master_df[['amfi_code', 'scheme_name']],how='left', on='amfi_code')
    ind_cat_metrics = ind_cat_metrics.rename({"consistency":"consistency_pct"},axis=1)
    scored_df = compute_composite_score(ind_cat_metrics)

    st.subheader("Fund Ranking")
    st.dataframe(scored_df[['scheme_name','aum_cr','aum_rank','consistency_score','downside_capture_score','upside_capture_score',
                            'sortino_score','sharpe_score','alpha_score','composite_score','rating']])