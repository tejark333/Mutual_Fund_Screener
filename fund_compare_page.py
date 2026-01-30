import streamlit as st
from analytics import *
import os
import pandas as pd

def fund_compare_page():
    
    # -----------------------------
    # LOAD MASTER DATA
    # -----------------------------
    master_df = load_amfi_master()
    aum_df = load_clean_aum()
    
    # -----------------------------
    # INPUTS
    # -----------------------------
    st.subheader("🎯 Analysis Configuration")
    
    typed_df = get_fund_types(master_df)
    
    st.subheader("🎯 Analysis Configuration")

    with st.container(border=True):
    
        # -----------------------------
        # ROW 1: Fund Type + Main Fund
        # -----------------------------
        # r1c2 = st.columns([6])
    
        eligible_funds = get_direct_growth_funds(master_df)
        
        primary_fund = st.selectbox(
            "Primary Fund",
            eligible_funds["scheme_name"].sort_values().tolist()
        )

        # -----------------------------
        # ROW 2: Funds to Compare
        # -----------------------------
        compare_funds = st.multiselect(
            "Funds to Compare",
            options=eligible_funds["scheme_name"].sort_values().tolist(),
            default=[],
            help="Select any number of funds for comparison"
        )
    
        # Ensure primary fund is not duplicated
        compare_funds = [
            f for f in compare_funds if f != primary_fund
        ]
    
        # -----------------------------
        # ROW 3: Analysis Settings
        # -----------------------------
        r3c1, r3c2, r3c3 = st.columns([2, 3, 3])
    
        with r3c1:
            rolling_window = st.selectbox(
                "Rolling Window (Years)",
                [3, 5]
            )
    
        with r3c2:
            benchmark_name = st.selectbox(
                "Benchmark (TRI)",
                list(TRI_BENCHMARKS.keys())
            )
    
        with r3c3:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True
            )

    
    # -----------------------------
    # MAIN ANALYTICS
    # -----------------------------
    if run:
        with st.spinner("Running analysis..."):
    
            benchmark_code = TRI_BENCHMARKS[benchmark_name]
            bench_df = read_tri(os.path.join("data", benchmark_code))
    
            # top_amfi_codes, top_df = get_top_n_by_aum(
            #     master_df,
            #     aum_df,
            #     category,
            #     top_n
            # )

            fund_name = primary_fund
            amfi_code = get_amfi_code(master_df, category="NA", fund_name=fund_name)
            nav_df = fetch_nav_data(amfi_code)

            
            top_amfi_codes = [get_amfi_code(master_df, category="NA", fund_name=fund_name) for fund_name in compare_funds]
            top_amfi_codes.append(amfi_code)
            top_df = add_aum_to_funds(master_df, aum_df, selected_codes=top_amfi_codes)
            
            END_DATE = pd.Timestamp("2025-12-31")
    
            # -----------------------------
            # TABS
            # -----------------------------
            tabs = st.tabs([
                "📌 Overview",
                "📈 Returns",
                "🔄 Rolling Analysis",
                "🧮 Consistency & Ranking",
                "⚠️ Risk Metrics",
                "📋 Detailed Tables"
            ])
    
    
    
            # =============================
            # TAB 1: OVERVIEW
            # =============================
            with tabs[0]:
                st.subheader("📌 Fund Snapshot")
    
                fund_aum = aum_df[aum_df["amfi_code"] == amfi_code]["aum_cr"]
    
                # category_df = master_df[master_df["category"] == category]
                # coverage = category_aum_coverage(aum_df, category_df, top_df)
    
                k1, k2 = st.columns(2)
                k1.metric("Inception Date", str(nav_df["date"].min().date()))
                k2.metric("Latest NAV Date", str(nav_df["date"].max().date()))

                k3, k4 = st.columns(2)
                if not fund_aum.empty:
                    k3.metric("AUM (₹ Cr)", f"{fund_aum.values[0]:,.0f}")
    
                # if coverage is not None:
                #     k4.metric(
                #         f"Top {top_n} Category AUM Coverage",
                #         f"{coverage:.1f}%"
                #     )
    
    
    
            # =============================
            # TAB 2: RETURNS
            # =============================
            with tabs[1]:
                st.subheader("📈 Performance Summary")
    
                st.plotly_chart(
                    plot_calendar_returns_multi_comp(
                        nav_df, top_amfi_codes, bench_df
                    ),
                    use_container_width=True
                )
    
                st.plotly_chart(
                    plot_cagr_summary_multi(
                        nav_df, top_amfi_codes, bench_df
                    ),
                    use_container_width=True
                )
    
            # =============================
            # TAB 3: ROLLING ANALYSIS
            # =============================
            with tabs[2]:
                st.subheader(f"🔄 {rolling_window}Y Rolling Returns")
    
                rr_df = rolling_returns(nav_df, rolling_window)
                col_name = f"{rolling_window}Y Rolling CAGR (%)"
    
                st.plotly_chart(
                    plot_rolling_returns_multi(
                        nav_df, top_amfi_codes, bench_df, rolling_window
                    ),
                    use_container_width=True
                )
    
                bw = best_worst_rolling_periods(rr_df, col_name)
    
                st.markdown("### 🏆 Best & Worst Periods")
                c1, c2 = st.columns(2)
                c1.dataframe(bw["best_period"].to_frame(), hide_index=True)
                c2.dataframe(bw["worst_period"].to_frame(), hide_index=True)
    
                percentiles = rolling_return_percentiles(rr_df, col_name)
                st.caption(
                    f"Median: {percentiles['50%']}% | "
                    f"25–75% Range: {percentiles['25%']}% – {percentiles['75%']}%"
                )
    
                st.plotly_chart(
                    plot_rolling_return_distribution(
                        rr_df, col_name, percentiles
                    ),
                    use_container_width=True
                )
    
            # =============================
            # TAB 4: CONSISTENCY & RANKING
            # =============================
            with tabs[3]:
                
                category_amfi_codes = sorted(top_amfi_codes)
    
                fund_rr = rolling_returns(nav_df, 3)
                cat_baseline = category_rolling_baseline(
                    category_amfi_codes, 3
                )
    
                scores = consistency_vs_category(
                    fund_rr, cat_baseline, 3
                )
    
                st.subheader("🧮 Consistency vs Category (3Y)")
                c1, c2 = st.columns(2)
                c1.metric("vs Category Median", f"{scores['median_consistency']}%")
                c2.metric("vs Category Mean", f"{scores['mean_consistency']}%")
    
                consistency_metric = 'median'
    
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
    
                st.plotly_chart(
                    plot_consistency_vs_aum_rank(cons_df, amfi_code),
                    use_container_width=True
                )
    
            # =============================
            # TAB 5: RISK METRICS
            # =============================
            with tabs[4]:
                fund_metrics = risk_metrics_monthly(
                    nav_df, bench_df, END_DATE
                )
    
                bench_metrics = risk_metrics_monthly(
                    bench_df, bench_df, END_DATE
                )
    
                ind_cat_metrics, category_metrics = (
                    category_risk_metrics_monthly(
                        top_amfi_codes, bench_df, END_DATE
                    )
                )
    
                risk_df = pd.DataFrame.from_dict(
                    {
                        fund_name: fund_metrics,
                        "Benchmark": bench_metrics,
                        "Category": category_metrics
                    },
                    orient="index"
                )
    
                st.subheader("⚠️ Risk Metrics (Monthly, Last 3 Years)")
                st.dataframe(
                    risk_df.style.format("{:.2f}"),
                    use_container_width=True
                )
    
            # =============================
            # TAB 6: DETAILED TABLES
            # =============================
            with tabs[5]:
                table_df = cons_df.sort_values("aum_cr", ascending=False)
                table_df["aum_rank"] = range(1, len(table_df) + 1)
        
                ind_cat_metrics["amfi_code"] = ind_cat_metrics["amfi_code"].astype(str)
    
                ind_cat_metrics = ind_cat_metrics.merge(
                    table_df[["amfi_code","aum_cr", "aum_rank", "consistency"]],
                    how="left",
                    on="amfi_code"
                ).merge(
                    master_df[["amfi_code", "scheme_name"]],
                    how="left",
                    on="amfi_code"
                )
    
                ind_cat_metrics = ind_cat_metrics.rename(
                    {"consistency": "consistency_pct"}, axis=1
                )
    
                scored_df = compute_composite_score(ind_cat_metrics)
    
                
                st.dataframe(
                    scored_df[
                        [
                            "scheme_name",
                            "aum_cr",
                            "aum_rank",
                            "consistency_score",
                            "downside_capture_score",
                            "upside_capture_score",
                            "sortino_score",
                            "sharpe_score",
                            "alpha_score",
                            "composite_score",
                            "rating",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True)
