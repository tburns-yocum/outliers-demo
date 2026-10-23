"""
Outlier Lab — Concord Decision Science Office Hours
One simulated A/B test, one whale/bot/logging-error toggle, six sections
that all react to the same underlying experiment.

Run with:  streamlit run outlier_lab.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ============================================================== constants
COLOR_BG = "#0b0f14"
COLOR_PANEL = "#131922"
COLOR_BORDER = "#23303b"
COLOR_TEXT = "#e7edf2"
COLOR_MUTED = "#7e93a3"
COLOR_CONTROL = "#5fa8d3"
COLOR_TREATMENT = "#34d6c4"
COLOR_MEAN = "#34d6c4"
COLOR_MEDIAN = "#b39dff"
COLOR_WHALE = "#f2a65a"
COLOR_BOT = "#e8636b"
COLOR_LOGERR = "#c9c15a"
COLOR_BAR = "#1c4a45"

TYPE_COLORS = {
    "normal": COLOR_CONTROL,
    "whale": COLOR_WHALE,
    "bot": COLOR_BOT,
    "logging_error": COLOR_LOGERR,
}
TYPE_LABELS = {
    "whale": "🐋 Whale purchase — real customer",
    "bot": "🤖 Bot traffic — fake / duplicated",
    "logging_error": "📝 Logging error — data bug",
}

SECTIONS = [
    "1 · Intro",
    "2 · What Is an Outlier",
    "3 · Why They Matter",
    "4 · Detecting Outliers",
    "5 · Handling Outliers",
    "6 · Reporting",
]


# ============================================================== simulation
@st.cache_data
def simulate_base(n_per_group, conversion_rate, mean_order, treatment_lift, seed):
    """Two-arm experiment, no outliers yet. Revenue = converted * order_value."""
    rng = np.random.default_rng(seed)
    rows = []
    for variant, lift in [("control", 0.0), ("treatment", treatment_lift)]:
        converted = rng.random(n_per_group) < conversion_rate
        order_values = np.exp(np.log(mean_order) + 0.5 * rng.standard_normal(n_per_group))
        order_values = order_values * (1 + lift)
        revenue = np.where(converted, order_values, 0.0)
        rows.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "converted": converted,
                    "order_value": order_values,
                    "revenue": revenue,
                    "type": "normal",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def inject_outliers(base_df, whale_on, whale_value, whale_count,
                     bots_on, bot_count, log_err_on, seed):
    rng = np.random.default_rng(seed + 999)
    df = base_df.copy()
    additions = []

    if whale_on and whale_count > 0:
        for _ in range(whale_count):
            additions.append(
                {"variant": "treatment", "converted": True,
                 "order_value": whale_value, "revenue": whale_value, "type": "whale"}
            )

    if bots_on and bot_count > 0:
        treat_converted = df[(df["variant"] == "treatment") & (df["converted"])]
        if len(treat_converted):
            dup = treat_converted.sample(n=bot_count, replace=True, random_state=int(seed) + 1)
            for _, r in dup.iterrows():
                additions.append(
                    {"variant": "treatment", "converted": True,
                     "order_value": r["order_value"], "revenue": r["revenue"], "type": "bot"}
                )

    if log_err_on:
        mean_order_est = df.loc[df["converted"], "order_value"].mean()
        bad_value = mean_order_est * 100  # e.g. a decimal-point logging bug
        additions.append(
            {"variant": "treatment", "converted": True,
             "order_value": bad_value, "revenue": bad_value, "type": "logging_error"}
        )

    if additions:
        df = pd.concat([df, pd.DataFrame(additions)], ignore_index=True)

    df["is_outlier"] = df["type"] != "normal"
    return df.reset_index(drop=True)


def welch_ttest(df):
    c = df.loc[df["variant"] == "control", "revenue"].to_numpy()
    t = df.loc[df["variant"] == "treatment", "revenue"].to_numpy()
    c_mean, t_mean = c.mean(), t.mean()
    tstat, pvalue = stats.ttest_ind(t, c, equal_var=False)
    lift = (t_mean - c_mean) / c_mean if c_mean else np.nan
    return {"control_mean": c_mean, "treatment_mean": t_mean, "lift": lift, "pvalue": pvalue}


def largest_customer_share(df):
    t = df[df["variant"] == "treatment"]
    total = t["revenue"].sum()
    if total <= 0 or len(t) == 0:
        return None
    top = t.loc[t["revenue"].idxmax()]
    return {"value": top["revenue"], "type": top["type"], "share": top["revenue"] / total}


def apply_technique(values, technique):
    """values: 1D np.array of order values. Returns (treated_values, note)."""
    if technique == "none" or len(values) == 0:
        return values, "No treatment applied."
    if technique == "winsorize":
        lo, hi = np.percentile(values, [1, 99])
        treated = np.clip(values, lo, hi)
        n_capped = int(np.sum(values != treated))
        return treated, f"{n_capped} point(s) capped to the 1st/99th percentile."
    if technique == "trim":
        m, s = values.mean(), values.std(ddof=1)
        z = np.abs((values - m) / s) if s > 0 else np.zeros_like(values)
        keep = z <= 3
        n_removed = int(np.sum(~keep))
        return values[keep], f"{n_removed} point(s) with |z| > 3 removed."
    return values, ""


def fmt_money(v):
    return f"${v:,.0f}"


# ============================================================== plotting
def revenue_histogram(df, value_col="order_value", title=""):
    fig = go.Figure()
    max_val = df[value_col].max() if len(df) else 1
    bins = np.linspace(0, max_val * 1.05, 40)

    for typ in ["normal", "whale", "bot", "logging_error"]:
        sub = df[df["type"] == typ]
        if len(sub) == 0:
            continue
        counts, edges = np.histogram(sub[value_col], bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_bar(
            x=centers, y=counts,
            marker_color=TYPE_COLORS[typ],
            name=typ.replace("_", " ").title(),
            opacity=0.9 if typ == "normal" else 1.0,
        )

    fig.update_layout(
        barmode="stack",
        height=320,
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font_color=COLOR_TEXT,
        margin=dict(l=10, r=10, t=30 if title else 10, b=10),
        title=title,
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(title="Order value ($)", gridcolor=COLOR_BORDER, tickprefix="$"),
        yaxis=dict(title="Count", gridcolor=COLOR_BORDER),
    )
    return fig


def revenue_boxplot(df, value_col="order_value"):
    fig = go.Figure()
    fig.add_box(
        y=df[value_col], name="Order value",
        marker_color=COLOR_CONTROL, boxpoints=False, fillcolor="#1c4a45",
        line_color=COLOR_CONTROL,
    )
    for typ in ["whale", "bot", "logging_error"]:
        sub = df[df["type"] == typ]
        if len(sub):
            fig.add_scatter(
                x=["Order value"] * len(sub), y=sub[value_col],
                mode="markers", marker=dict(color=TYPE_COLORS[typ], size=10, symbol="circle-open",
                                             line=dict(width=2)),
                name=typ.replace("_", " ").title(),
            )
    fig.update_layout(
        height=340, plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG, font_color=COLOR_TEXT,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Order value ($)", gridcolor=COLOR_BORDER, tickprefix="$"),
        xaxis=dict(gridcolor=COLOR_BORDER),
    )
    return fig


# ============================================================== app shell
st.set_page_config(page_title="Outlier Lab", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: {COLOR_PANEL}; }}
    .ol-eyebrow {{ text-transform: uppercase; letter-spacing: 0.14em; font-size: 11px;
        color: {COLOR_MUTED}; font-weight: 600; }}
    .ol-title {{ font-size: 28px; font-weight: 700; margin: 4px 0 10px; }}
    .ol-caption {{ font-size: 14px; color: #d3ac7a; font-weight: 500; margin: 8px 0 16px; }}
    .ol-callout {{ background: {COLOR_PANEL}; border-left: 3px solid {COLOR_WHALE};
        padding: 12px 16px; border-radius: 4px; font-size: 15px; margin: 12px 0; }}
    .ol-card {{ background: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER};
        border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }}
    div[data-testid="stMetric"] {{ background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 14px; }}
    div[data-testid="stMetricLabel"] {{ color: {COLOR_MUTED} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- sidebar: experiment + outlier controls ----
st.sidebar.markdown("### Experiment setup")
n_per_group = st.sidebar.slider("Visitors per variant", 200, 5000, 1000, step=100)
conversion_rate = st.sidebar.slider("Baseline conversion rate", 0.02, 0.30, 0.10, step=0.01)
mean_order = st.sidebar.slider("Average order value ($)", 20, 250, 70, step=5)
treatment_lift = st.sidebar.slider("True treatment lift", -0.10, 0.20, 0.05, step=0.01)
seed = st.sidebar.number_input("Random seed", 1, 100_000_000, 20260729)

st.sidebar.markdown("---")
st.sidebar.markdown("### Outliers in the treatment arm")
whale_on = st.sidebar.checkbox("🐋 Add whale purchase(s)")
whale_value = st.sidebar.slider("Whale order size ($)", 500, 100_000, 25_000, step=500, disabled=not whale_on)
whale_count = st.sidebar.slider("Number of whales", 1, 5, 1, disabled=not whale_on)

bots_on = st.sidebar.checkbox("🤖 Add bot traffic")
bot_count = st.sidebar.slider("Number of bot orders", 1, 100, 20, disabled=not bots_on)

log_err_on = st.sidebar.checkbox("📝 Add a logging error")

st.sidebar.markdown("---")
section = st.sidebar.radio("Section", SECTIONS, label_visibility="collapsed")

# ---- build the (single, shared) experiment ----
base_df = simulate_base(n_per_group, conversion_rate, mean_order, treatment_lift, seed)
df = inject_outliers(base_df, whale_on, whale_value, whale_count, bots_on, bot_count, log_err_on, seed)
stats_now = welch_ttest(df)
converted_df = df[df["converted"]].copy()

st.markdown('<div class="ol-eyebrow">Concord — Lunch and Learn</div>', unsafe_allow_html=True)
st.markdown('<div class="ol-title">Outlier Lab</div>', unsafe_allow_html=True)


# ============================================================== Section 1
def page_cold_open():
    st.markdown("#### Could one customer change your experiment?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control · Revenue/Visitor", fmt_money(stats_now["control_mean"]))
    c2.metric("Treatment · Revenue/Visitor", fmt_money(stats_now["treatment_mean"]))
    c3.metric("Lift", f"{stats_now['lift']*100:+.1f}%")
    sig = "✅" if stats_now["pvalue"] < 0.05 else "—"
    c4.metric("p-value", f"{stats_now['pvalue']:.3f}", sig)

    st.plotly_chart(revenue_histogram(converted_df, title="Order value distribution (converted visitors)"),
                     use_container_width=True)

    top = largest_customer_share(df)
    if top and top["type"] != "normal":
        st.markdown(
            f'<div class="ol-callout">The largest treatment order was '
            f'<b>{fmt_money(top["value"])}</b> ({TYPE_LABELS[top["type"]]}), contributing '
            f'<b>{top["share"]*100:.0f}%</b> of total treatment revenue.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Toggle a whale, bot, or logging error in the sidebar to see the lift and p-value move.")


# ============================================================== Section 2
def page_what_is_outlier():
    st.markdown("#### What is an outlier?")
    st.write(
        "Not every extreme value is the same problem. The fix depends on **why** the point "
        "is extreme — not just how far it sits from the mean."
    )
    cols = st.columns(3)
    descriptions = {
        "whale": ("🐋 Legitimate extreme value",
                  "A real customer placed a real, unusually large order. The business value is real — "
                  "it just has outsized *leverage* on a small-sample average."),
        "bot": ("🤖 Contamination",
                "Fake or duplicated traffic that never should have been in the sample. "
                "It doesn't represent real customer behavior."),
        "logging_error": ("📝 Data error",
                           "A measurement or pipeline bug (e.g. a misplaced decimal, double-counted event). "
                           "The value never actually happened."),
    }
    for col, (key, (head, body)) in zip(cols, descriptions.items()):
        with col:
            st.markdown(f'<div class="ol-card"><b>{head}</b><br><span style="color:{COLOR_MUTED};'
                        f'font-size:13px;">{body}</span></div>', unsafe_allow_html=True)

    active = df[df["is_outlier"]]
    if len(active):
        st.markdown("**Currently active in your simulated experiment:**")
        show = active[["type", "order_value"]].copy()
        show["type"] = show["type"].map(lambda t: TYPE_LABELS.get(t, t))
        show = show.rename(columns={"type": "Cause", "order_value": "Value ($)"})
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No outliers active — turn one on in the sidebar to populate this table.")


# ============================================================== Section 3
def page_why_matter():
    st.markdown("#### Why they matter: mean vs. median")
    st.write(
        "A textbook illustration first, then the same effect in your live experiment."
    )
    textbook = [35, 42, 51, 61, 44, 39, 48]
    whale_demo = st.slider("Add one whale order to the textbook example ($)", 0, 5000, 0, step=100)
    demo_vals = textbook + ([whale_demo] if whale_demo > 0 else [])
    tc1, tc2 = st.columns(2)
    tc1.metric("Mean", fmt_money(np.mean(demo_vals)))
    tc2.metric("Median", fmt_money(np.median(demo_vals)))
    st.caption(f"Base orders: {textbook}" + (f"  +  one order at ${whale_demo:,}" if whale_demo else ""))

    st.markdown("---")
    st.markdown("**Your live experiment (treatment, converted orders):**")
    vals = converted_df.loc[converted_df["variant"] == "treatment", "order_value"].to_numpy()
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Mean", fmt_money(vals.mean()) if len(vals) else "—")
    lc2.metric("Median", fmt_money(np.median(vals)) if len(vals) else "—")
    lc3.metric("N", len(vals))
    st.caption("Toggle whale/bot/logging error in the sidebar — watch the mean move and the median barely budge.")


# ============================================================== Section 4
def page_detecting():
    st.markdown("#### Detecting outliers")
    tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "Boxplot", "Z-score", "IQR"])
    vals = converted_df["order_value"].to_numpy()

    with tab1:
        st.plotly_chart(revenue_histogram(converted_df), use_container_width=True)

    with tab2:
        st.plotly_chart(revenue_boxplot(converted_df), use_container_width=True)
        st.caption("Points outside the whiskers are colored by cause, not just distance.")

    with tab3:
        threshold = st.select_slider("z-score threshold", options=[2, 2.5, 3, 4], value=3)
        m, s = vals.mean(), vals.std(ddof=1)
        z = np.abs((vals - m) / s) if s > 0 else np.zeros_like(vals)
        flagged = converted_df[z > threshold]
        st.metric(f"Points with |z| > {threshold}", len(flagged))
        if len(flagged):
            st.dataframe(flagged[["variant", "order_value", "type"]], use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="ol-callout">⚠️ If the underlying data are skewed, the mean and standard '
            "deviation used to compute z are already distorted by the outlier — z-score can miss or "
            "misclassify exactly the points you're looking for.</div>",
            unsafe_allow_html=True,
        )

    with tab4:
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = converted_df[(converted_df["order_value"] < lo) | (converted_df["order_value"] > hi)]
        c1, c2, c3 = st.columns(3)
        c1.metric("Q1 / Q3", f"{fmt_money(q1)} / {fmt_money(q3)}")
        c2.metric("Upper fence", fmt_money(hi))
        c3.metric("Flagged points", len(flagged))
        if len(flagged):
            st.dataframe(flagged[["variant", "order_value", "type"]], use_container_width=True, hide_index=True)


# ============================================================== Section 5
def page_handling():
    st.markdown("#### Handling outliers — the main event")
    st.write("Same experiment, three treatments. Watch the p-value.")

    technique = st.radio("Method", ["None", "Winsorize (1st–99th pctile)", "Trim (|z| > 3)"], horizontal=True)
    technique_id = {"None": "none", "Winsorize (1st–99th pctile)": "winsorize", "Trim (|z| > 3)": "trim"}[technique]

    treat_vals = converted_df.loc[converted_df["variant"] == "treatment", "order_value"].to_numpy()
    treated_vals, note = apply_technique(treat_vals, technique_id)
    st.caption(note)

    # rebuild a revenue-equivalent dataframe for the t-test: treatment revenue swapped in,
    # non-converted visitors (revenue 0) stay as-is, control untouched.
    control_rev = df.loc[df["variant"] == "control", "revenue"].to_numpy()
    n_treat_total = (df["variant"] == "treatment").sum()
    n_nonconverted_treat = n_treat_total - len(treat_vals)
    treat_rev = np.concatenate([treated_vals, np.zeros(max(0, n_nonconverted_treat))])

    tstat, pvalue = stats.ttest_ind(treat_rev, control_rev, equal_var=False)
    lift = (treat_rev.mean() - control_rev.mean()) / control_rev.mean() if control_rev.mean() else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N (order values)", len(treated_vals), delta=f"{len(treated_vals) - len(treat_vals)}" if len(treated_vals) != len(treat_vals) else None)
    c2.metric("Mean order value", fmt_money(treated_vals.mean()) if len(treated_vals) else "—")
    c3.metric("Lift", f"{lift*100:+.1f}%")
    sig = "✅ significant" if pvalue < 0.05 else "not significant"
    c4.metric("p-value", f"{pvalue:.3f}", sig)

    treated_df_display = converted_df[converted_df["variant"] == "treatment"].copy()
    if len(treated_vals) == len(treat_vals):
        treated_df_display["order_value"] = treated_vals
    else:
        # trimming changed length; just show the treated array's own histogram
        treated_df_display = pd.DataFrame({"order_value": treated_vals, "type": "normal"})
    st.plotly_chart(revenue_histogram(treated_df_display, title="Treatment distribution after treatment applied"),
                     use_container_width=True)


# ============================================================== Section 6
def page_reporting():
    st.markdown("#### Reporting & the decision framework")

    st.markdown("**What caused this outlier?**")
    cause = st.radio(
        "Cause", ["Real enterprise customer", "Bot / fake traffic", "Logging bug", "Duplicate event", "Unsure"],
        horizontal=True, label_visibility="collapsed",
    )
    recs = {
        "Real enterprise customer": ("Winsorize", "The value happened and is real. Cap its leverage, keep its signal."),
        "Bot / fake traffic": ("Remove", "This observation doesn't represent real customer behavior."),
        "Logging bug": ("Remove", "The value never actually occurred — it's a measurement artifact."),
        "Duplicate event": ("Remove", "De-duplicate rather than treat as a legitimate extreme value."),
        "Unsure": ("Investigate before deciding", "Don't guess — trace the record back to its source system first."),
    }
    action, reason = recs[cause]
    st.markdown(f'<div class="ol-callout"><b>Recommended action: {action}</b><br>'
                f'<span style="color:{COLOR_MUTED}">{reason}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**How the same result reads with vs. without disclosure**")
    hide = st.toggle("Hide methodology")
    lift_pct = stats_now["lift"] * 100
    if hide:
        st.markdown(
            f'<div class="ol-card">Revenue lift: <b>{lift_pct:+.1f}%</b><br>'
            f'p-value: <b>{stats_now["pvalue"]:.3f}</b></div>',
            unsafe_allow_html=True,
        )
        st.caption("This looks like a clean, ready-to-ship result. Would you ship it?")
    else:
        method_note = "No outlier handling applied." if not df["is_outlier"].any() else (
            f"{int(df['is_outlier'].sum())} outlier point(s) present, untreated in this figure."
        )
        st.markdown(
            f'<div class="ol-card">Revenue lift: <b>{lift_pct:+.1f}%</b><br>'
            f'p-value: <b>{stats_now["pvalue"]:.3f}</b><br>'
            f'Outlier handling: <b>{method_note}</b></div>',
            unsafe_allow_html=True,
        )
        st.caption("Same number — now you can tell whether to trust it.")

    st.markdown("---")
    st.markdown("### Takeaway")
    st.write("One customer, one bot, or one bad log line shouldn't get a unilateral vote on whether your experiment shipped. Decide your outlier policy *before* you look at the results — and always report it.")


PAGES = {
    SECTIONS[0]: page_cold_open,
    SECTIONS[1]: page_what_is_outlier,
    SECTIONS[2]: page_why_matter,
    SECTIONS[3]: page_detecting,
    SECTIONS[4]: page_handling,
    SECTIONS[5]: page_reporting,
}

PAGES[section]()
