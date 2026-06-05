"""
Portfolio Analyser specific analytics.

Unlike the Fund Analysis / Fund Compare pages — where every fund is rated
against ONE shared benchmark — a portfolio holds funds from different
categories, so EACH fund must be rated against:

  * the benchmark appropriate to ITS OWN category, and
  * the leading PEER COHORT of its own category (NOT the other funds in the
    portfolio).

The peer cohort for a category = the top funds by AUM whose cumulative AUM
reaches an 80% coverage threshold of that category's total AUM, with a minimum
floor so thin categories still rank fairly. Every metric score (consistency,
upside/downside capture, sortino, sharpe, alpha) is a percentile rank WITHIN
that category cohort, so a fund's rating does not change based on which other
funds happen to be in the portfolio.

Import order matters: import this AFTER `from analytics import *` so the shared
helpers (risk_metrics_monthly, category_risk_metrics_monthly,
category_rolling_baseline, consistency_for_top_n, compute_composite_score,
get_*_end_dates, RATING_ORDER, score_label, read_tri, TRI_BENCHMARKS,
get_funds_by_category, ...) are available.
"""

from analytics import *
import os
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# CATEGORY -> BENCHMARK MAPPING
# ---------------------------------------------------------------------------
# Ordered (keyword, benchmark_name). First keyword found as a substring of the
# lower-cased category wins, so ORDER MATTERS: "Large & Mid Cap Fund" contains
# "mid cap", so the "large & mid" rule must be tested before plain "mid cap".
# Indices without local TRI data (Nifty LargeMidcap 250, Nifty500 Multicap
# 50:25:25, Nifty Dividend Opportunities 50) map to the Nifty 500 TRI fallback.
FALLBACK_BENCHMARK = "Nifty 50 TRI"

CATEGORY_BENCHMARK_RULES = [
    ("large & mid", "Nifty 500 TRI"),          # Nifty LargeMidcap 250 -> fallback
    ("large and mid", "Nifty 500 TRI"),        # alt spelling -> fallback
    ("large cap", "Nifty 100 TRI"),
    ("mid cap", "Nifty Midcap 150 TRI"),
    ("small cap", "Nifty Smallcap 250 TRI"),
    ("flexi cap", "Nifty 500 TRI"),
    ("multi cap", "Nifty 500 TRI"),            # Nifty500 Multicap 50:25:25 -> fallback
    ("elss", "Nifty 500 TRI"),
    ("tax saver", "Nifty 500 TRI"),
    ("focused", "Nifty 500 TRI"),
    ("value", "Nifty 500 TRI"),
    ("contra", "Nifty 500 TRI"),
    ("dividend yield", "Nifty 500 TRI"),       # Nifty Dividend Opportunities 50 -> fallback
]

# Peer-cohort sizing.
AUM_COVERAGE = 0.80     # take top funds until cumulative AUM reaches 80%
MIN_COHORT_FUNDS = 5    # ...but never fewer than this (when the category has them)


def benchmark_for_category(category):
    """Map an AMFI category string to a TRI benchmark name (always in TRI_BENCHMARKS)."""
    if not category:
        return FALLBACK_BENCHMARK

    cat = str(category).lower()
    for keyword, bench in CATEGORY_BENCHMARK_RULES:
        if keyword in cat:
            return bench if bench in TRI_BENCHMARKS else FALLBACK_BENCHMARK

    return FALLBACK_BENCHMARK


def get_category_for_code(master_df, amfi_code):
    """Look up the AMFI category for a fund code."""
    row = master_df[master_df["amfi_code"].astype(str) == str(amfi_code)]
    if row.empty:
        return ""
    return row["category"].values[0]


def resolve_fund_benchmarks(master_df, codes):
    """Return {amfi_code (str) -> benchmark_name} for every fund in the portfolio."""
    return {
        str(code): benchmark_for_category(get_category_for_code(master_df, code))
        for code in codes
    }


@st.cache_data(ttl=24 * 60 * 60)
def get_benchmark_df(benchmark_name):
    """Load a benchmark TRI by its display name (cached)."""
    code = TRI_BENCHMARKS.get(benchmark_name, TRI_BENCHMARKS[FALLBACK_BENCHMARK])
    return read_tri(os.path.join("data", code))


# ---------------------------------------------------------------------------
# UPLOADED-FILE FUND RESOLUTION
# ---------------------------------------------------------------------------
def _read_fund_list(uploaded_file):
    """Flatten an uploaded csv / xlsx / txt into a list of fund-name strings."""
    name = getattr(uploaded_file, "name", "").lower()

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    content = uploaded_file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    return [ln.strip() for ln in content.splitlines() if ln.strip()]


def resolve_uploaded_funds(uploaded_file, master_df):
    """Match uploaded fund names against eligible Direct-Growth schemes.

    Returns (matched_scheme_names, unmatched_inputs). Exact (case-insensitive)
    first, then a forgiving substring match.
    """
    eligible = get_direct_growth_funds(master_df)
    names = eligible["scheme_name"].tolist()
    lower_map = {n.lower().strip(): n for n in names}

    raw = _read_fund_list(uploaded_file)

    matched, unmatched = [], []
    for item in raw:
        key = str(item).lower().strip()
        if not key:
            continue
        if key in lower_map:
            matched.append(lower_map[key])
            continue
        hits = [orig for low, orig in lower_map.items() if key in low or low in key]
        if hits:
            matched.append(hits[0])
        else:
            unmatched.append(str(item).strip())

    seen = set()
    matched_unique = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            matched_unique.append(m)

    return matched_unique, unmatched


def portfolio_funds_with_aum(master_df, aum_df, codes):
    """Attach AUM to the portfolio funds via a LEFT merge (keeps funds w/o AUM)."""
    codes = [str(c) for c in codes]

    df = master_df[master_df["amfi_code"].astype(str).isin(codes)].copy()
    df["amfi_code"] = df["amfi_code"].astype(str)

    aum = aum_df.copy()
    aum["amfi_code"] = aum["amfi_code"].astype(str)

    df = df.merge(aum, on="amfi_code", how="left")
    if "aum_cr" in df.columns:
        df["aum_cr"] = df["aum_cr"].fillna(0)
    else:
        df["aum_cr"] = 0

    return df.sort_values("aum_cr", ascending=False)


# ---------------------------------------------------------------------------
# PEER COHORT (top funds covering 80% of category AUM, min floor)
# ---------------------------------------------------------------------------
def get_category_cohort_by_aum(
    master_df,
    aum_df,
    category,
    coverage=AUM_COVERAGE,
    min_funds=MIN_COHORT_FUNDS,
    include_codes=None,
):
    """Top Direct-Growth funds of a category by AUM, up to `coverage` cumulative
    AUM (min `min_funds`). `include_codes` are force-added so the portfolio's own
    funds are always rankable even if they fall below the cohort line.
    """
    cat = get_funds_by_category(master_df, category).copy()  # Direct + Growth, exact category
    cat["amfi_code"] = cat["amfi_code"].astype(str)

    aum = aum_df.copy()
    aum["amfi_code"] = aum["amfi_code"].astype(str)

    cat = cat.merge(aum, on="amfi_code", how="inner")
    if cat.empty:
        return cat

    cat = cat.sort_values("aum_cr", ascending=False).reset_index(drop=True)

    total = cat["aum_cr"].sum()
    if total > 0:
        cum_share = cat["aum_cr"].cumsum() / total
        n = int((cum_share < coverage).sum()) + 1   # include the fund that crosses the threshold
    else:
        n = len(cat)

    n = max(n, min_funds)
    n = min(n, 10)
    n = min(n, len(cat))
    cohort = cat.head(n).copy()

    if include_codes:
        inc = [str(c) for c in include_codes]
        extra = cat[
            cat["amfi_code"].isin(inc)
            & ~cat["amfi_code"].isin(cohort["amfi_code"])
        ]
        if not extra.empty:
            cohort = pd.concat([cohort, extra], ignore_index=True)

    return cohort


def _group_codes_by_category(master_df, codes):
    by_cat = {}
    for c in codes:
        c = str(c)
        cat = get_category_for_code(master_df, c)
        by_cat.setdefault(cat, []).append(c)
    return by_cat


# ---------------------------------------------------------------------------
# PER-CATEGORY RATING CONTEXT
# ---------------------------------------------------------------------------
# The cohort, its benchmark, and the cohort consistency are AUM/history based
# and do NOT depend on the rating cutoff date, so we compute them once per
# category and reuse them across every historical cutoff. Only the monthly risk
# metrics (which depend on the cutoff) are recomputed per date.
def _prepare_category_context(
    master_df, aum_df, category, port_codes_in_cat, window_years, coverage, min_funds
):
    cohort = get_category_cohort_by_aum(
        master_df, aum_df, category, coverage, min_funds, include_codes=port_codes_in_cat
    )
    if cohort.empty or len(cohort) < 2:
        return None

    cohort_codes = sorted(cohort["amfi_code"].astype(str).tolist())

    bench_name = benchmark_for_category(category)
    bench_df = get_benchmark_df(bench_name)

    # consistency of each cohort fund vs the cohort's rolling-return baseline
    baseline = category_rolling_baseline(cohort_codes, window_years)
    cons = consistency_for_top_n(cohort, baseline, window_years)
    if cons.empty:
        return None
    cons["amfi_code"] = cons["amfi_code"].astype(str)

    table = cohort.merge(cons[["amfi_code", "consistency"]], on="amfi_code", how="left")
    table = table.sort_values("aum_cr", ascending=False).reset_index(drop=True)
    table["aum_rank"] = range(1, len(table) + 1)

    return {
        "category": category,
        "bench_name": bench_name,
        "bench_df": bench_df,
        "cohort_codes": cohort_codes,
        "table": table[["amfi_code", "aum_cr", "aum_rank", "consistency", "scheme_name"]],
        "port_codes": [str(c) for c in port_codes_in_cat],
    }


def build_portfolio_contexts(
    master_df, aum_df, codes, window_years,
    coverage=AUM_COVERAGE, min_funds=MIN_COHORT_FUNDS,
):
    """One rating context per category represented in the portfolio."""
    by_cat = _group_codes_by_category(master_df, codes)
    contexts = []
    for cat, port_codes in by_cat.items():
        ctx = _prepare_category_context(
            master_df, aum_df, cat, port_codes, window_years, coverage, min_funds
        )
        if ctx is not None:
            contexts.append(ctx)
    return contexts


def _score_context(ctx, end_date):
    """Score the whole cohort at `end_date` (percentile within the cohort)."""
    ind, _ = category_risk_metrics_monthly(ctx["cohort_codes"], ctx["bench_df"], end_date)
    if ind is None or len(ind) < 2:
        return None

    ind["amfi_code"] = ind["amfi_code"].astype(str)

    ind = (
        ind.merge(ctx["table"], how="left", on="amfi_code")
        .rename({"consistency": "consistency_pct"}, axis=1)
    )

    # need every metric used by the composite; drop funds lacking history
    ind = ind.dropna(
        subset=[
            "consistency_pct", "downside_capture", "upside_capture",
            "sortino", "sharpe", "alpha",
        ]
    )
    if len(ind) < 2:
        return None

    scored = compute_composite_score(ind)
    scored["benchmark"] = ctx["bench_name"]
    scored["category"] = ctx["category"]
    return scored


# ---------------------------------------------------------------------------
# CURRENT & HISTORICAL RATING (portfolio funds only, scored vs their cohort)
# ---------------------------------------------------------------------------
def portfolio_current_rating_table(contexts, end_date):
    """Full current-rating rows for the portfolio funds (component + composite)."""
    frames = []
    for ctx in contexts:
        scored = _score_context(ctx, end_date)
        if scored is None:
            continue
        keep = scored[scored["amfi_code"].isin(ctx["port_codes"])]
        if not keep.empty:
            frames.append(keep)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def portfolio_rating_snapshot(contexts, cutoff_date):
    """Compact rating row per portfolio fund at a cutoff date (for heatmaps)."""
    empty = pd.DataFrame(
        columns=["scheme_name", "period_end", "composite_score", "rating", "rating_level"]
    )

    scored = portfolio_current_rating_table(contexts, cutoff_date)
    if scored is None or scored.empty:
        return empty

    s = scored.copy()
    s["rating"] = s["composite_score"].apply(score_label)
    s["rating_level"] = s["rating"].map(RATING_ORDER)
    s["period_end"] = cutoff_date

    return s[["scheme_name", "period_end", "composite_score", "rating", "rating_level"]]


def build_portfolio_rating_history(contexts, end_date, freq="M", n=6):
    """Stack rating snapshots across month / quarter / year ends."""
    if freq == "M":
        dates = get_month_end_dates(end_date, n)
    elif freq == "Q":
        dates = get_quarter_end_dates(end_date, n)
    else:
        dates = get_year_end_dates(end_date, n)

    frames = [portfolio_rating_snapshot(contexts, d) for d in dates]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame(
            columns=["scheme_name", "period_end", "composite_score", "rating", "rating_level"]
        )
    return pd.concat(frames, ignore_index=True)
