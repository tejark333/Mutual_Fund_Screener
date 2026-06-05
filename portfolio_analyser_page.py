import streamlit as st
from analytics import *
from analytics_portfolio import *
import pandas as pd


def portfolio_analyser_page():

    # -----------------------------
    # LOAD MASTER DATA
    # -----------------------------
    master_df = load_amfi_master()
    aum_df = load_clean_aum()

    eligible_funds = get_direct_growth_funds(master_df)

    END_DATE = pd.Timestamp("2025-12-31")

    # -----------------------------
    # INPUTS
    # -----------------------------
    st.subheader("🎯 Portfolio Configuration")

    with st.container(border=True):

        input_mode = st.radio(
            "How do you want to add your funds?",
            ["Select from list", "Upload a file"],
            horizontal=True,
        )

        selected_names = []
        unmatched = []

        if input_mode == "Select from list":
            selected_names = st.multiselect(
                "Funds in your portfolio",
                options=eligible_funds["scheme_name"].sort_values().tolist(),
                default=[],
                help="Select at least 2 funds",
            )
        else:
            uploaded = st.file_uploader(
                "Upload your fund list (CSV, Excel, or TXT — one fund per row)",
                type=["csv", "xlsx", "xls", "txt"],
            )
            if uploaded is not None:
                selected_names, unmatched = resolve_uploaded_funds(
                    uploaded, master_df
                )
                st.success(f"Matched {len(selected_names)} fund(s) from your file.")
                if unmatched:
                    st.warning(
                        "Couldn't match: " + ", ".join(unmatched[:20])
                        + ("..." if len(unmatched) > 20 else "")
                    )

        r1, r2 = st.columns([2, 3])

        with r1:
            rolling_window = st.selectbox("Rolling Window (Years)", [3, 5])

        with r2:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True,
            )

    # -----------------------------
    # MAIN ANALYTICS
    # -----------------------------
    if not run:
        return

    if len(selected_names) < 2:
        st.warning(
            "Please add at least 2 funds — the rating is relative and needs "
            "more than one fund to compare."
        )
        return

    with st.spinner("Analysing your portfolio..."):

        # ---- resolve codes + per-fund benchmarks ----
        codes = sorted(
            str(get_amfi_code(master_df, category="NA", fund_name=name))
            for name in selected_names
        )
        code_to_bench = resolve_fund_benchmarks(master_df, codes)

        # ---- portfolio funds + AUM (for the Fund Selection display only) ----
        top_df = portfolio_funds_with_aum(master_df, aum_df, codes)

        # ---- per-category rating contexts ----
        # Each fund is rated against the leading peer cohort of ITS OWN
        # category (top funds covering 80% of category AUM, min 5), NOT against
        # the other funds in the portfolio. Built once and reused across every
        # historical cutoff.
        contexts = build_portfolio_contexts(
            master_df, aum_df, codes, rolling_window
        )

        # -----------------------------
        # TABS
        # -----------------------------
        tabs = st.tabs([
            "🗂️ Fund Selection",
            "⭐ Current Rating",
            "🕒 Historical Rating",
        ])

        # =============================
        # TAB 1: FUND SELECTION
        # =============================
        with tabs[0]:
            st.subheader("🗂️ Funds & Benchmarks")

            sel_df = top_df[["scheme_name", "category", "aum_cr"]].copy()
            sel_df["benchmark"] = sel_df.index.map(
                lambda i: code_to_bench.get(str(top_df.loc[i, "amfi_code"]), "")
            )
            sel_df = sel_df.rename(
                columns={
                    "scheme_name": "Fund",
                    "category": "Category",
                    "benchmark": "Benchmark",
                    "aum_cr": "AUM (₹ Cr)",
                }
            )

            st.dataframe(
                sel_df,
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Each fund is rated against the benchmark mapped from its "
                "category. Where the exact index isn't available, Nifty 500 "
                "TRI is used as the fallback."
            )

        # =============================
        # TAB 2: CURRENT RATING
        # =============================
        with tabs[1]:
            st.subheader("⭐ Current Rating")

            scored_df = portfolio_current_rating_table(contexts, END_DATE)

            if scored_df is None or scored_df.empty:
                st.warning(
                    "Not enough funds with 3+ years of history to produce a "
                    "relative rating."
                )
            else:
                show = scored_df.rename(
                    columns={
                        "scheme_name": "Fund",
                        "category": "Category",
                        "benchmark": "Benchmark",
                        "aum_cr": "AUM (₹ Cr)",
                    }
                )
                st.dataframe(
                    show[
                        [
                            "Fund",
                            "Category",
                            "Benchmark",
                            "AUM (₹ Cr)",
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
                    use_container_width=True,
                )
                st.caption(
                    "Each score is a percentile rank of the fund within the "
                    "leading peer cohort of its own category — independent of "
                    "the other funds in your portfolio."
                )

        # =============================
        # TAB 3: HISTORICAL RATING
        # =============================
        with tabs[2]:
            st.subheader("🕒 Historical Rating")

            monthly_df = build_portfolio_rating_history(
                contexts, END_DATE, freq="M", n=6,
            )
            if not monthly_df.empty:
                st.plotly_chart(
                    plot_rating_heatmap(
                        monthly_df,
                        "Portfolio Rating Heatmap – Monthly (Last 6 Months)",
                    ),
                    use_container_width=True,
                )

            quarterly_df = build_portfolio_rating_history(
                contexts, END_DATE, freq="Q", n=16,
            )
            if not quarterly_df.empty:
                st.plotly_chart(
                    plot_rating_heatmap(
                        quarterly_df,
                        "Portfolio Rating Heatmap – Quarterly (Last 16 Quarters)",
                    ),
                    use_container_width=True,
                )

            yearly_df = build_portfolio_rating_history(
                contexts, END_DATE, freq="Y", n=10,
            )
            if not yearly_df.empty:
                st.plotly_chart(
                    plot_rating_heatmap(
                        yearly_df,
                        "Portfolio Rating Heatmap – Yearly (Last 10 Years)",
                    ),
                    use_container_width=True,
                )
