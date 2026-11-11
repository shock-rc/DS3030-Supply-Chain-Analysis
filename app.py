
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from scipy import stats

st.set_page_config(
    page_title="Supply Chain Financial & Delay Analysis Dashboard",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("DS3030 : Supply Chain Financial & Delay Analysis Dashboard")
st.caption("The data used for the dashboard can be referenced at : https://example.com/")

DATA_FILE = "Supply_Chain_Shipment_Data_Cleaned.csv"


@st.cache_data
def load_data():
    d = pd.read_csv(DATA_FILE)
    d.drop(columns = 'freight_status', inplace=True)

    return d


@st.cache_data
def calculate_pca(data):
    features = [
        "weight (kilograms)",
        "line item quantity",
        "line item value",
        "pack price",
        "unit price",
        "freight cost (usd)",
        "freight_per_kg",
    ]
    x = data[features].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    pca = PCA()
    z = pca.fit_transform(xs)
    result = data.loc[x.index].copy()
    result["PC1"] = z[:, 0]
    result["PC2"] = z[:, 1]
    result["PC3"] = z[:, 2]
    return result, pca, features


@st.cache_data
def calculate_clusters(data, k):
    features = [
        "weight (kilograms)",
        "line item quantity",
        "line item value",
        "freight cost (usd)",
        "freight_per_kg",
        "freight_pct_of_value",
    ]
    x = data[features].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(xs)

    out = data.loc[x.index].copy()
    out["cluster"] = labels
    sil = silhouette_score(xs, labels) if k > 1 and len(np.unique(labels)) > 1 else np.nan
    return out, model, scaler, features, sil


@st.cache_data
def association_rules(data, min_support=0.03, min_confidence=0.30):
    """
    Lightweight Apriori-style pair mining for dashboard use.
    Each shipment becomes a transaction of categorical tokens.
    We mine frequent 1-item and 2-item combinations and derive rules.
    """
    cols = ["product group", "shipment mode", "fulfill via", "vendor inco term"]
    work = data[cols].fillna("Unknown").astype(str)

    transactions = []
    for _, row in work.iterrows():
        items = {f"{c}={row[c]}" for c in cols}
        transactions.append(items)

    n = len(transactions)
    item_count = {}
    pair_count = {}

    for items in transactions:
        for item in items:
            item_count[item] = item_count.get(item, 0) + 1
        items_list = sorted(items)
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                pair = (items_list[i], items_list[j])
                pair_count[pair] = pair_count.get(pair, 0) + 1

    rows = []
    for (a, b), count in pair_count.items():
        support = count / n
        if support < min_support:
            continue
        conf_ab = count / item_count[a]
        conf_ba = count / item_count[b]
        # P(B) and P(A) are estimated directly from item counts.
        lift_ab = conf_ab / (item_count[b] / n)
        lift_ba = conf_ba / (item_count[a] / n)

        if conf_ab >= min_confidence:
            rows.append({
                "antecedent": a, "consequent": b,
                "support": support, "confidence": conf_ab, "lift": lift_ab
            })
        if conf_ba >= min_confidence:
            rows.append({
                "antecedent": b, "consequent": a,
                "support": support, "confidence": conf_ba, "lift": lift_ba
            })

    if not rows:
        return pd.DataFrame(
            columns=["antecedent", "consequent", "support", "confidence", "lift"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["lift", "confidence"], ascending=False)
        .drop_duplicates(["antecedent", "consequent"])
        .reset_index(drop=True)
    )


df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Dashboard Controls")

countries = st.sidebar.multiselect(
    "Destination country",
    sorted(df["country"].dropna().unique()),
    default=[],
)

modes = st.sidebar.multiselect(
    "Shipment mode",
    sorted(df["shipment mode"].dropna().unique()),
    default=[],
)

products = st.sidebar.multiselect(
    "Product group",
    sorted(df["product group"].dropna().unique()),
    default=[],
)

managed = st.sidebar.multiselect(
    "Managed by",
    sorted(df["managed by"].dropna().unique()),
    default=[],
)

filtered = df.copy()
if countries:
    filtered = filtered[filtered["country"].isin(countries)]
if modes:
    filtered = filtered[filtered["shipment mode"].isin(modes)]
if products:
    filtered = filtered[filtered["product group"].isin(products)]
if managed:
    filtered = filtered[filtered["managed by"].isin(managed)]


page = st.pills(
    "Analysis module",
    [
        "Overview",
        "Data Processing",
        "Statistics",
        "Delay Analysis",
        "Supply Map",
        "PCA",
        "Clustering",
        "Association Rules",
        "Insights & Hypothests Testing",
    ],
    default="Data Processing",
    label_visibility="collapsed"
)

# ---------------- Overview ----------------
if page == "Overview":
    # ---------------- Main KPIs ----------------
    st.markdown("### Executive Overview")
    c1, c2, c3, c4, c5 = st.columns(5)

    total_shipments = len(filtered)
    total_value = filtered["line item value"].sum()
    freight = filtered["freight cost (usd)"].sum()
    avg_distance_note = "Distance requires a geographic lookup; see Insights."
    late_pct = (filtered["delivery_status"].eq("Late").mean() * 100) if total_shipments else 0
    data_completeness = 100 * filtered[pd.read_csv("Supply_Chain_Shipment_Data.csv").columns].notna().mean().mean()

    c1.metric("Shipments", f"{total_shipments:,}")
    c2.metric("Line-item value", f"${total_value:,.0f}")
    c3.metric("Freight cost", f"${freight:,.0f}")
    c4.metric("Late deliveries", f"{late_pct:.1f}%")
    c5.metric("Data completeness", f"{data_completeness:.1f}%")

    st.divider()
    st.subheader("Brief Overview of the Supply Chain Data")

    a, b = st.columns(2)

    with a:
        cost_mode = (
            filtered.groupby("shipment mode", dropna=False)["freight cost (usd)"]
            .sum().reset_index()
        )
        cost_mode["shipment mode"] = cost_mode["shipment mode"].fillna("Unknown")
        fig = px.bar(
            cost_mode, x="shipment mode", y="freight cost (usd)",
            title="Freight cost by shipment mode",
            labels={"freight cost (usd)": "Freight cost (USD)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with b:
        delivery = (
            filtered["delivery_status"].value_counts()
            .reindex(["Early", "On Time", "Late", "Unknown"], fill_value=0)
            .rename_axis("status").reset_index(name="shipments")
        )
        fig = px.bar(
            delivery, x="status", y="shipments",
            title="Delivery performance",
            labels={"shipments": "Shipments"}
        )
        st.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)

    with c:
        scatter = filtered.dropna(subset=["line item value", "freight cost (usd)"]).copy()
        fig = px.scatter(
            scatter, x="line item value", y="freight cost (usd)",
            color="shipment mode",
            hover_data=["country", "product group", "vendor"],
            title="Shipment value vs Line Item Value",
            labels={
                "line item value": "Line item value (USD)",
                "freight cost (usd)": "Freight cost (USD)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with d:
        top_countries = (
            filtered.groupby("country")["freight cost (usd)"]
            .sum().nlargest(10).reset_index().sort_values("freight cost (usd)", ascending=True)
        )
        fig = px.bar(
            top_countries, x="freight cost (usd)", y="country",
            orientation="h", title="Top 10 destination countries by freight cost",
            labels={"freight cost (usd)": "Freight cost (USD)"}
        )
        st.plotly_chart(fig, use_container_width=True)




# ---------------- Data Processing ----------------
elif page == "Data Processing":
    st.subheader("Data Cleaning & Summary of Data")

    raw = pd.read_csv("Supply_Chain_Shipment_Data.csv")
    numeric_cleaned = df

    before_missing = int(raw.isna().sum().sum())
    after_missing = int(numeric_cleaned[raw.columns].isna().sum().sum())

    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.metric("Raw rows", f"{len(raw):,}")
    q2.metric("Raw columns", f"{raw.shape[1]:,}")
    q3.metric("Columns after pre-processing", f"{numeric_cleaned.shape[1]:,}")
    q4.metric("Missing cells in Raw Data", f"{before_missing:,}")
    q5.metric("Missing cells after cleaning", f"{after_missing:,}")
    q6.metric("Duplicate rows", f"{raw.duplicated().sum():,}")

    _, _, _, note_col, _ = st.columns([1, 1, 1, 2, 1])

    with note_col:
        st.markdown(
            """
            <div style="background-color: #ffe6e6; border-left: 4px solid #ff4b4b; padding: 6px 10px; color: #cc0000; font-size: 0.8em; border-radius: 4px; margin-top: 5px;">
                <strong>Note:</strong> Only NaN and NaT counts are shown here using df.isna()
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Data challenge → analytical response")
    challenges = pd.DataFrame({
        "Challenge": [
            "Data given as reference to other rows",
            "Numerical Information were stored as strings with invalid values",
            "Values in weights and freight cost columns were MCAR",
            "Date Information were stored as strings which could not be easily used for calculations"
        ],
        "Response": [
            "Imputed the information by referencing said rows",
            "All Numerical Information were convered to numeric values",
            "Median Imputation was performed on the column as sufficient correlations wasn't found with other columns",
            "Converted all date columns to datetime format and made a new delivery delay column and delay status column"
        ],
    })
    st.dataframe(challenges, use_container_width=True, hide_index=True)


    missing = (
        numeric_cleaned.isna().sum().sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing"})
    )
    missing["missing_pct"] = 100 * missing["missing"] / len(numeric_cleaned)

    fig = px.bar(
        missing.head(12).sort_values("missing"),
        x="missing_pct", y="column", orientation="h",
        title="Columns with the most missing values",
        labels={"missing_pct": "Missing (%)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    

# ---------------- Delay Analysis ----------------
elif page == "Delay Analysis":

    st.subheader("Delivery Delay Analysis")
    st.caption(
        "Analysis of delivery performance using scheduled and actual client delivery dates."
    )

    # ---------------------------------------------------------
    # Basic delay statistics
    # ---------------------------------------------------------

    delay_data = filtered.copy()

    delay_data["delivery_delay_days"] = pd.to_numeric(
        delay_data["delivery_delay_days"],
        errors="coerce"
    )

    total = len(delay_data)

    late = (delay_data["delivery_status"] == "Late").sum()
    on_time = (delay_data["delivery_status"] == "On Time").sum()
    early = (delay_data["delivery_status"] == "Early").sum()

    late_pct = 100 * late / total if total else 0
    on_time_pct = 100 * on_time / total if total else 0
    early_pct = 100 * early / total if total else 0

    avg_delay = delay_data.loc[
        delay_data["delivery_delay_days"] > 0,
        "delivery_delay_days"
    ].mean()

    max_delay = delay_data["delivery_delay_days"].max()

    # ---------------------------------------------------------
    # KPI cards
    # ---------------------------------------------------------

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Late deliveries",
        f"{late:,}",
        f"{late_pct:.1f}%"
    )

    k2.metric(
        "On-time deliveries",
        f"{on_time:,}",
        f"{on_time_pct:.1f}%"
    )

    k3.metric(
        "Early deliveries",
        f"{early:,}",
        f"{early_pct:.1f}%"
    )

    k4.metric(
        "Average delay",
        f"{avg_delay:.1f} days" if pd.notna(avg_delay) else "N/A"
    )

    k5.metric(
        "Maximum delay",
        f"{max_delay:.0f} days" if pd.notna(max_delay) else "N/A"
    )

    st.divider()

    # ---------------------------------------------------------
    # Delivery status distribution + delay distribution
    # ---------------------------------------------------------

    a, b = st.columns(2)

    with a:

        status_order = ["Early", "On Time", "Late"]

        status_df = (
            delay_data["delivery_status"]
            .value_counts()
            .reindex(status_order, fill_value=0)
            .rename_axis("Status")
            .reset_index(name="Shipments")
        )

        fig = px.bar(
            status_df,
            x="Status",
            y="Shipments",
            text="Shipments",
            title="Delivery Performance",
            labels={"Shipments": "Number of shipments"}
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with b:

        delay_dist = delay_data.dropna(
            subset=["delivery_delay_days"]
        ).copy()

        fig = px.histogram(
            delay_dist,
            x="delivery_delay_days",
            nbins=40,
            title="Distribution of Delivery Delay",
            labels={
                "delivery_delay_days": "Delay / early delivery (days)"
            }
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            annotation_text="On-time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------------------------------
    # Delay by shipment mode
    # ---------------------------------------------------------

    st.markdown("### Delivery Delay Distribution by Shipment Mode")

    kde_data = filtered.copy()

    # Ensure delay is numeric
    kde_data["delivery_delay_days"] = pd.to_numeric(
        kde_data["delivery_delay_days"],
        errors="coerce"
    )

    fig = go.Figure()

    modes = (
        kde_data["shipment mode"]
        .dropna()
        .unique()
    )

    for mode in modes:

        mode_delay = (
            kde_data.loc[
                kde_data["shipment mode"] == mode,
                "delivery_delay_days"
            ]
            .dropna()
            .values
        )

        # KDE needs more than one observation
        if len(mode_delay) < 2:
            continue

        # KDE fails if all values are identical
        if np.std(mode_delay) == 0:
            continue

        kde = gaussian_kde(mode_delay)

        # Use a common x-axis
        x = np.linspace(
            kde_data["delivery_delay_days"].quantile(0.02),
            kde_data["delivery_delay_days"].quantile(0.98),
            1000
        )

        y = kde(x)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=str(mode),
                fill="tozeroy",
                opacity=0.25
            )
        )

    fig.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="On Time",
        annotation_position="top"
    )

    fig.update_layout(
        title="Delivery Delay Modelled via Kernel Density Estimation (KDE)",
        xaxis_title="Delivery Delay (days)",
        yaxis_title="Density",
        hovermode="x unified",
        height=550,
        legend_title="Shipment Mode",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # Delay by destination country
    # ---------------------------------------------------------

    st.markdown("### Destination-Level Delay Analysis")

    country_delay = (
        delay_data
        .groupby("country", dropna=False)
        .agg(
            shipments=("id", "count"),
            late_shipments=(
                "delivery_status",
                lambda x: (x == "Late").sum()
            ),
            avg_delay=("delivery_delay_days", "mean")
        )
        .reset_index()
    )

    country_delay["late_pct"] = (
        100
        * country_delay["late_shipments"]
        / country_delay["shipments"]
    )

    country_delay["country"] = (
        country_delay["country"]
        .fillna("Unknown")
    )

    # Only show countries with enough shipments
    country_delay_plot = country_delay[
        country_delay["shipments"] >= 10
    ].copy()

    c, d = st.columns(2)

    with c:

        top_late = (
            country_delay_plot
            .sort_values("late_pct", ascending=False)
            .head(15)
        )

        fig = px.bar(
            top_late.sort_values("late_pct"),
            x="late_pct",
            y="country",
            orientation="h",
            title="Countries with Highest Late-Delivery Rate",
            labels={
                "late_pct": "Late deliveries (%)",
                "country": "Destination country"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with d:

        top_delay = (
            country_delay_plot
            .sort_values("avg_delay", ascending=False)
            .head(15)
        )

        fig = px.bar(
            top_delay.sort_values("avg_delay"),
            x="avg_delay",
            y="country",
            orientation="h",
            title="Countries with Highest Average Delay",
            labels={
                "avg_delay": "Average delay (days)",
                "country": "Destination country"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------------------------------
    # Delay trend over time
    # ---------------------------------------------------------

    st.markdown("### Delivery Delay Trend")

    trend = delay_data.dropna(
        subset=["delivered to client date"]
    ).copy()

    trend["delivery_month"] = (
        pd.to_datetime(
            trend["delivered to client date"],
            errors="coerce"
        )
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        trend
        .groupby("delivery_month")
        .agg(
            shipments=("id", "count"),
            avg_delay=("delivery_delay_days", "mean"),
            late_rate=(
                "delivery_status",
                lambda x: 100 * (x == "Late").mean()
            )
        )
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="delivery_month",
        y="late_rate",
        markers=True,
        title="Monthly Late-Delivery Rate",
        labels={
            "delivery_month": "Delivery month",
            "late_rate": "Late deliveries (%)"
        }
    )

    fig.add_hline(
        y=late_pct,
        line_dash="dash",
        annotation_text=f"Overall: {late_pct:.1f}%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("""
    ### Historical Context

    - **2009–2012:** HIV/ARV supply chains faced funding, procurement, manufacturing, and transportation disruptions, contributing to increased shipment delays.
    - **2007–2012:** Generic ARV manufacturers delivered only **50–70% of orders on time**, indicating significant supply-chain instability during this period.
    - The 2008 financial crisis and subsequent global economic downturn further strained supply chains, affecting the timely delivery of essential medications.
    - These factors provide historical context for the **delay spike observed in the dataset**.

    **Source:** [Strategies to Reduce Risks in ARV Supply Chains](https://pmc.ncbi.nlm.nih.gov/articles/PMC4307856/)
    """)


    # ---------------------------------------------------------
    # Most delayed shipments
    # ---------------------------------------------------------

    st.markdown("### Most Delayed Shipments")

    delayed_shipments = (
        delay_data[
            delay_data["delivery_status"] == "Late"
        ]
        .sort_values(
            "delivery_delay_days",
            ascending=False
        )
        .head(20)
    )

    display_cols = [
        "id",
        "country",
        "product group",
        "shipment mode",
        "vendor",
        "scheduled delivery date",
        "delivered to client date",
        "delivery_delay_days",
        "line item value",
        "freight cost (usd)"
    ]

    display_cols = [
        c for c in display_cols
        if c in delayed_shipments.columns
    ]

    st.dataframe(
        delayed_shipments[display_cols],
        use_container_width=True,
        hide_index=True
    )


# ---------------- Statistics ----------------
elif page == "Statistics":
    st.subheader("Statistical Profile & Cost Efficiency")

    num_cols = [
        "line item quantity", "line item value", "pack price", "unit price",
        "weight (kilograms)", "freight cost (usd)",
        "freight_per_kg", "freight_per_unit", "freight_pct_of_value"
    ]
    stats = filtered[num_cols].describe().T
    stats["missing"] = filtered[num_cols].isna().sum()
    st.dataframe(stats.round(3), use_container_width=True)

    a, b = st.columns(2)
    with a:
        hist = filtered.dropna(subset=["freight cost (usd)"])
        fig = px.histogram(
            hist, x="freight cost (usd)", nbins=40,
            title="Freight-cost distribution",
            labels={"freight cost (usd)": "Freight cost (USD)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with b:
        box = filtered.dropna(subset=["freight cost (usd)", "shipment mode"])
        fig = px.box(
            box, x="shipment mode", y="freight cost (usd)",
            title="Freight-cost spread by shipment mode",
            labels={"freight cost (usd)": "Freight cost (USD)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### High-cost-per-unit shipments")
    efficiency = filtered.dropna(subset=["freight_per_unit"]).copy()
    efficiency = efficiency.sort_values("freight_per_unit", ascending=False).head(15)
    show_cols = [
        "id", "country", "product group", "shipment mode",
        "line item quantity", "freight cost (usd)", "freight_per_unit",
        "vendor"
    ]
    st.dataframe(
        efficiency[show_cols].round(3),
        use_container_width=True,
        hide_index=True
    )






# ---------------- PCA ----------------
elif page == "PCA":
    # List of the 9 original feature columns used for PCA
    pca_features = [
        "line item quantity",
        "line item value",
        "unit price",
        "weight (kilograms)",
        "freight cost (usd)",
        "line item insurance (usd)",
        "freight_per_kg",
        "freight_pct_of_value",
        "delivery_delay_days"
    ]

    # Create an expander matching your image's title style
    with st.expander("Features used for PCA", expanded=True):
        # st.multiselect naturally acts as interactive tags with 'x' remove buttons
        pos_cols = st.multiselect(
            label="Features",
            options=pca_features,
            default=pca_features,  # All selected by default like the image
            label_visibility="collapsed" # Hides the redundant label to look clean like a tag container
        )

    # ============================================================
    # FEATURE SELECTION
    # ============================================================

    d = filtered.copy()

    # Remove invalid zero/negative values
    d = d[
        (d["weight (kilograms)"] > 0) &
        (d["line item value"] > 0)
    ].copy()

    X = pd.DataFrame(index=d.index)

    # Log transform positive, heavily skewed variables
    for c in pos_cols:
        X["log_" + c] = np.log1p(d[c])

    # Signed log for delivery delay because it can be negative
    X["signed_log_delay"] = (
        np.sign(d["delivery_delay_days"]) *
        np.log1p(d["delivery_delay_days"].abs())
    )

    # Replace invalid values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Median imputation
    X = X.fillna(X.median())

    # Remove zero-variance columns
    X = X.loc[:, X.std() > 1e-8]

    # ============================================================
    # STANDARDIZATION
    # ============================================================

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ------------------------------------------------------------
    # Explained variance
    # ------------------------------------------------------------

    pca_full = PCA().fit(Xs)

    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)

    components = list(range(1, len(evr) + 1))

    # Plot explained variance
    fig_ev = go.Figure()

    fig_ev.add_trace(
        go.Bar(
            x=components,
            y=evr * 100,
            name="Individual Variance"
        )
    )

    fig_ev.add_trace(
        go.Scatter(
            x=components,
            y=cum * 100,
            mode="lines+markers",
            name="Cumulative Variance",
        )
    )

    fig_ev.add_hline(
        y=80,
        line_dash="dash",
        line_color="gray",
        annotation_text="80%",
        annotation_position="top left",
    )

    fig_ev.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Individual Variance (%)",
        yaxis2=dict(
            title="Cumulative Variance (%)",
            overlaying="y",
            side="right",
            range=[0, 105]
        ),
        height=550,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_ev,
        use_container_width=True
    )


    # ------------------------------------------------------------
    # PCA with 3 components
    # ------------------------------------------------------------

    pca = PCA(n_components=3).fit(Xs)

    P = pca.transform(Xs)

    v = pca.explained_variance_ratio_ * 100


    # ============================================================
    # PCA LOADINGS
    # ============================================================

    st.markdown("### PCA Component Loadings")

    loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=X.columns,
        columns=["PC1", "PC2", "PC3"]
    )

    st.dataframe(
        loadings.round(3),
        use_container_width=True
    )


    # ============================================================
    # CLUSTERING
    # ============================================================

    st.subheader("K-Means Clustering")

    # ------------------------------------------------------------
    # Evaluate k = 2 to 10
    # ------------------------------------------------------------

    ks = range(2, 11)

    inertia = []
    sil = []

    for k in ks:

        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        ).fit(Xs)

        inertia.append(km.inertia_)

        sil.append(
            silhouette_score(
                Xs,
                km.labels_,
                sample_size=5000,
                random_state=42
            )
        )

    best_k = list(ks)[int(np.argmax(sil))]

    st.write(
        f"**Best k according to silhouette score:** {best_k}"
    )


    # ============================================================
    # ELBOW + SILHOUETTE PLOTS
    # ============================================================

    col1, col2 = st.columns(2)


    # ------------------------------------------------------------
    # Elbow curve
    # ------------------------------------------------------------

    fig_elbow = go.Figure()

    fig_elbow.add_trace(
        go.Scatter(
            x=list(ks),
            y=inertia,
            mode="lines+markers",
            name="Inertia"
        )
    )

    fig_elbow.update_layout(
        title="Elbow Curve",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        height=500
    )

    with col1:
        st.plotly_chart(
            fig_elbow,
            use_container_width=True
        )


    # ------------------------------------------------------------
    # Silhouette score
    # ------------------------------------------------------------

    fig_sil = go.Figure()

    fig_sil.add_trace(
        go.Scatter(
            x=list(ks),
            y=sil,
            mode="lines+markers",
            name="Silhouette Score"
        )
    )

    fig_sil.add_vline(
        x=best_k,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Best k = {best_k}",
        annotation_position="top"
    )

    fig_sil.update_layout(
        title="Silhouette Score vs k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        height=500
    )

    with col2:
        st.plotly_chart(
            fig_sil,
            use_container_width=True
        )


    # ============================================================
    # FINAL K-MEANS MODEL
    # ============================================================

    km = KMeans(
        n_clusters=best_k,
        n_init=10,
        random_state=42
    ).fit(Xs)

    labels = km.labels_

    # Project cluster centroids into PCA space
    cent = pca.transform(km.cluster_centers_)


    # ============================================================
    # CLUSTERS IN PCA SPACE
    # ============================================================

    st.markdown("### K-Means Clusters in PCA Space")

    cluster_plot = pd.DataFrame(
        {
            "PC1": P[:, 0],
            "PC2": P[:, 1],
            "Cluster": labels.astype(str),
            "Line Item Value": d["line item value"].values,
            "Weight (kg)": d["weight (kilograms)"].values,
            "Delivery Delay (days)": d["delivery_delay_days"].values
        },
        index=d.index
    )

    fig_clusters = px.scatter(
        cluster_plot,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=[
            "Line Item Value",
            "Weight (kg)",
            "Delivery Delay (days)"
        ],
        labels={
            "PC1": f"PC1 ({v[0]:.1f}% variance)",
            "PC2": f"PC2 ({v[1]:.1f}% variance)"
        },
        title=f"K-Means (k={best_k}) in PCA Space"
    )

    fig_clusters.update_traces(
        marker=dict(
            size=6,
            opacity=0.4
        )
    )

    # Add cluster centroids
    for c in range(best_k):

        fig_clusters.add_trace(
            go.Scatter(
                x=[cent[c, 0]],
                y=[cent[c, 1]],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=18,
                    color="black",
                    line=dict(width=2)
                ),
                name=f"Centroid {c}",
                showlegend=True
            )
        )

    fig_clusters.update_layout(
        height=700
    )

    st.plotly_chart(
        fig_clusters,
        use_container_width=True
    )


    # ============================================================
    # CLUSTER SIZE
    # ============================================================

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_sizes.columns = ["Cluster", "Observations"]

    cluster_sizes["Cluster"] = (
        cluster_sizes["Cluster"]
        .astype(str)
    )

    fig_size = px.bar(
        cluster_sizes,
        x="Cluster",
        y="Observations",
        text="Observations",
        title="Number of Shipment Line Items per Cluster"
    )

    fig_size.update_traces(
        textposition="outside"
    )

    fig_size.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_size,
        use_container_width=True
    )


    # ============================================================
    # CLUSTER PROFILE
    # ============================================================

    st.markdown("### Cluster Profile — Median Values")

    clustered_data = d.copy()
    clustered_data["cluster"] = labels

    profile = (
        clustered_data
        .groupby("cluster")[pos_cols + ["delivery_delay_days"]]
        .median()
        .round(2)
        .T
    )

    st.dataframe(
        profile,
        use_container_width=True
    )


    # ============================================================
    # OPTIONAL: CLUSTER PROFILE HEATMAP
    # ============================================================

    # Calculate standardized feature means per cluster
    profile_z = (
        pd.DataFrame(
            Xs,
            columns=X.columns,
            index=d.index
        )
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
    )

    fig_heatmap = px.imshow(
        profile_z,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Mean Standardized Feature Values by Cluster"
    )

    fig_heatmap.update_layout(
        height=500,
        xaxis_title="Features",
        yaxis_title="Cluster"
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )



# ---------------- Clustering ----------------
elif page == "Clustering":
    # ============================================================
    # CLUSTERING ANALYSIS
    # Uses `filtered` as the input dataframe
    # ============================================================
    # List of the 9 original feature columns used for PCA
    pca_features = [
        "line item quantity",
        "line item value",
        "unit price",
        "weight (kilograms)",
        "freight cost (usd)",
        "line item insurance (usd)",
        "freight_per_kg",
        "freight_pct_of_value",
        "delivery_delay_days"
    ]

    # Create an expander matching your image's title style
    with st.expander("Features used for Clustering", expanded=True):
        # st.multiselect naturally acts as interactive tags with 'x' remove buttons
        pos_cols = st.multiselect(
            label="Features",
            options=pca_features,
            default=pca_features,  # All selected by default like the image
            label_visibility="collapsed" # Hides the redundant label to look clean like a tag container
        )

    # ============================================================
    # 1. PREPARE DATA
    # ============================================================

    d = filtered.copy()

    # Keep valid observations
    d = d[
        (d["weight (kilograms)"] > 0) &
        (d["line item value"] > 0)
    ].copy()


    # ------------------------------------------------------------
    # Filler-value detection
    # ------------------------------------------------------------

    w_mode = d["weight (kilograms)"].mode()[0]
    f_mode = d["freight cost (usd)"].mode()[0]

    d["weight_is_filler"] = (
        d["weight (kilograms)"] == w_mode
    )

    d["freight_is_filler"] = (
        d["freight cost (usd)"] == f_mode
    )

    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================

    X = pd.DataFrame(index=d.index)

    for c in pos_cols:
        X["log_" + c] = np.log1p(d[c])

    # Signed log because delivery delay can be negative
    X["signed_log_delay"] = (
        np.sign(d["delivery_delay_days"]) *
        np.log1p(d["delivery_delay_days"].abs())
    )

    # Clean invalid values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Median imputation
    X = X.fillna(X.median())

    # Remove zero-variance columns
    X = X.loc[:, X.std() > 1e-8]


    # ============================================================
    # STANDARDIZATION
    # ============================================================

    scaler = StandardScaler()

    Xs = scaler.fit_transform(X)


    # ============================================================
    # PCA FOR VISUALIZATION
    # ============================================================

    pca = PCA(n_components=2)

    P = pca.fit_transform(Xs)

    v = pca.explained_variance_ratio_ * 100


    # ============================================================
    # K-MEANS CLUSTERING
    # ============================================================

    # Use k = 3 as in your original analysis
    K = st.slider(
    "Number of Clusters (k)",
    min_value=1,
    max_value=10,
    value=10,
    step=1
    )

    km = KMeans(
        n_clusters=K,
        n_init=10,
        random_state=42
    )

    d["cluster"] = km.fit_predict(Xs)

    labels = d["cluster"].values


    # ============================================================
    # CLUSTER SIZES
    # ============================================================

    cl = sorted(d["cluster"].unique())

    size = (
        d["cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_size_df = pd.DataFrame({
        "Cluster": [f"Cluster {c}" for c in size.index],
        "Observations": size.values,
        "Percentage": (size.values / len(d) * 100).round(1)
    })


    fig_size = px.bar(
        cluster_size_df,
        x="Cluster",
        y="Observations",
        text="Observations",
        title="Number of Shipment Line Items per Cluster"
    )

    fig_size.update_traces(
        textposition="outside"
    )

    fig_size.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_size,
        use_container_width=True
    )


    # ============================================================
    # 2. PCA CLUSTER VISUALIZATION
    # ============================================================

    st.subheader("K-Means Clusters in PCA Space")

    # Spatial aggregation ONLY for visualization
    # Clustering itself was performed on every observation.
    nb = 45

    bx = pd.cut(
        P[:, 0],
        nb,
        labels=False
    )

    by = pd.cut(
        P[:, 1],
        nb,
        labels=False
    )

    b = pd.DataFrame({
        "x": P[:, 0],
        "y": P[:, 1],
        "bin": bx * nb + by,
        "cluster": d["cluster"].values
    })

    agg = (
        b.groupby("bin")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            n=("x", "size"),
            cluster=(
                "cluster",
                lambda s: s.mode()[0]
            ),
            purity=(
                "cluster",
                lambda s:
                s.value_counts(normalize=True).iloc[0]
            )
        )
        .reset_index(drop=True)
    )

    # Centroids projected into PCA space
    cent = pca.transform(
        km.cluster_centers_
    )

    # Plot aggregated points
    fig_pca_cluster = go.Figure()

    for c in cl:

        a = agg[
            agg["cluster"] == c
        ]

        fig_pca_cluster.add_trace(
            go.Scatter(
                x=a["x"],
                y=a["y"],
                mode="markers",
                name=f"Cluster {c} (n={size[c]:,})",
                marker=dict(
                    size=8,
                    opacity=0.8,
                    line=dict(
                        width=0.5
                    )
                ),
                customdata=np.column_stack([
                    a["n"],
                    a["purity"]
                ]),
                hovertemplate=(
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "Observations: %{customdata[0]:,.0f}<br>"
                    "Cluster purity: %{customdata[1]:.2%}"
                    "<extra></extra>"
                )
            )
        )

    # Centroids
    fig_pca_cluster.add_trace(
        go.Scatter(
            x=cent[:, 0],
            y=cent[:, 1],
            mode="markers",
            name="Centroids",
            marker=dict(
                symbol="x",
                size=18,
                color="black",
                line=dict(
                    width=2
                )
            )
        )
    )

    fig_pca_cluster.update_layout(
        title=(
            f"K-Means (k={K}) in PCA Space"
        ),
        xaxis_title=f"PC1 ({v[0]:.1f}% variance)",
        yaxis_title=f"PC2 ({v[1]:.1f}% variance)",
        height=700,
        hovermode="closest"
    )

    st.plotly_chart(
        fig_pca_cluster,
        use_container_width=True
    )

    st.caption(
        "Each plotted point represents a spatial grid bin. "
        "Point size indicates the number of shipment observations "
        "in that bin. Clustering was performed using all observations."
    )


    # ============================================================
    # 3. STANDARDIZED CLUSTER PROFILE HEATMAP
    # ============================================================

    st.subheader("Standardized Cluster Profile")

    z = (
        pd.DataFrame(
            Xs,
            columns=X.columns,
            index=d.index
        )
        .groupby(d["cluster"])
        .mean()
    )

    fig_heatmap = px.imshow(
        z,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.5,
        zmax=1.5,
        color_continuous_midpoint=0,
        labels={
            "x": "Feature",
            "y": "Cluster",
            "color": "Mean z-score"
        },
        title="Cluster Profile (Standardized, Log-Scale Features)"
    )

    fig_heatmap.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )


    # ============================================================
    # 4. BOXPLOTS OF KEY VARIABLES
    # ============================================================

    st.subheader("Key Variables by Cluster")

    key = [
        "line item value",
        "weight (kilograms)",
        "freight cost (usd)",
        "freight_pct_of_value",
        "freight_per_kg",
        "line item quantity"
    ]

    selected_box_col = st.selectbox(
        "Select variable",
        key
    )

    # Clip only for plotting, same idea as original code
    box_data = d[
        ["cluster", selected_box_col]
    ].copy()

    box_data[selected_box_col] = (
        box_data[selected_box_col]
        .clip(lower=1e-2)
    )

    fig_box = px.box(
        box_data,
        x="cluster",
        y=selected_box_col,
        color="cluster",
        points=False,
        log_y=True,
        title=f"{selected_box_col} by Cluster"
    )

    fig_box.update_layout(
        xaxis_title="Cluster",
        yaxis_title=selected_box_col,
        height=550,
        showlegend=False
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

    st.caption(
        "Logarithmic y-axis is used because these variables are strongly right-skewed. "
        "Outliers are hidden only for visualization."
    )


    # ============================================================
    # 5. CATEGORICAL COMPOSITION
    # ============================================================

    st.subheader("Categorical Composition by Cluster")

    cat_plot_cols = [
        "shipment mode",
        "vendor inco term",
        "first line designation",
        "delivery_status"
    ]

    selected_cat = st.selectbox(
        "Select categorical variable",
        cat_plot_cols
    )

    cat_pct = (
        pd.crosstab(
            d["cluster"],
            d[selected_cat].astype(str),
            normalize="index"
        ) * 100
    )

    cat_pct = (
        cat_pct
        .reset_index()
        .melt(
            id_vars="cluster",
            var_name=selected_cat,
            value_name="percentage"
        )
    )

    fig_cat = px.bar(
        cat_pct,
        x="cluster",
        y="percentage",
        color=selected_cat,
        barmode="stack",
        title=f"{selected_cat} by Cluster",
        labels={
            "cluster": "Cluster",
            "percentage": "% of cluster"
        }
    )

    fig_cat.update_layout(
        height=600,
        yaxis=dict(
            range=[0, 100],
            title="% of Cluster"
        )
    )

    st.plotly_chart(
        fig_cat,
        use_container_width=True
    )


    # ============================================================
    # 6. COUNTRY COMPOSITION
    # ============================================================

    st.subheader("Country and Year Composition")

    # Top 8 countries
    top_c = (
        d["country"]
        .value_counts()
        .index[:8]
    )

    country_pct = (
        pd.crosstab(
            d["country"],
            d["cluster"],
            normalize="index"
        ) * 100
    )

    country_pct = (
        country_pct
        .loc[top_c]
        .reset_index()
        .melt(
            id_vars="country",
            var_name="cluster",
            value_name="percentage"
        )
    )

    country_pct["cluster"] = (
        "Cluster " +
        country_pct["cluster"].astype(str)
    )

    fig_country = px.bar(
        country_pct,
        x="country",
        y="percentage",
        color="cluster",
        barmode="stack",
        title="Top-8 Countries: % of Shipments per Cluster",
        labels={
            "percentage": "%"
        }
    )

    fig_country.update_layout(
        height=600,
        yaxis=dict(
            range=[0, 100]
        ),
        xaxis_tickangle=-40
    )

    st.plotly_chart(
        fig_country,
        use_container_width=True
    )


    # ============================================================
    # YEAR COMPOSITION
    # ============================================================

    year_pct = (
        pd.crosstab(
            d["delivery_year"],
            d["cluster"],
            normalize="index"
        ) * 100
    )

    year_pct = (
        year_pct
        .reset_index()
        .melt(
            id_vars="delivery_year",
            var_name="cluster",
            value_name="percentage"
        )
    )

    year_pct["cluster"] = (
        "Cluster " +
        year_pct["cluster"].astype(str)
    )

    fig_year = px.bar(
        year_pct,
        x="delivery_year",
        y="percentage",
        color="cluster",
        barmode="stack",
        title="Delivery Year: % of Shipments per Cluster",
        labels={
            "delivery_year": "Delivery Year",
            "percentage": "%"
        }
    )

    fig_year.update_layout(
        height=600,
        yaxis=dict(
            range=[0, 100]
        )
    )

    st.plotly_chart(
        fig_year,
        use_container_width=True
    )


    # ============================================================
    # 7. VALUE VS WEIGHT
    # ============================================================

    value_weight = d[
        [
            "line item value",
            "weight (kilograms)",
            "cluster",
            "country",
            "product group",
            "delivery_delay_days"
        ]
    ].copy()

    # Remove invalid values for log-scale plot
    value_weight = value_weight[
        (value_weight["line item value"] > 0) &
        (value_weight["weight (kilograms)"] > 0)
    ]

    value_weight["cluster"] = (
        "Cluster " +
        value_weight["cluster"].astype(str)
    )

    fig_value_weight = px.scatter(
        value_weight,
        x="line item value",
        y="weight (kilograms)",
        color="cluster",
        hover_data=[
            "country",
            "product group",
            "delivery_delay_days"
        ],
        log_x=True,
        log_y=True,
        opacity=0.25,
        title="Value vs Weight by Cluster",
        labels={
            "line item value": "Line Item Value (USD)",
            "weight (kilograms)": "Weight (kg)"
        }
    )

    fig_value_weight.update_traces(
        marker=dict(
            size=6
        )
    )

    fig_value_weight.update_layout(
        height=650
    )

    st.plotly_chart(
        fig_value_weight,
        use_container_width=True
    )

    st.caption(
        "Log scales are used for both axes because shipment value and weight "
        "are strongly right-skewed."
    )


    # ============================================================
    # 8. LATE DELIVERY RATE
    # ============================================================

    d["late"] = (
        d["delivery_delay_days"] > 0
    )

    late_rate = (
        d.groupby("cluster")["late"]
        .mean()
        .mul(100)
        .reset_index()
    )

    late_rate.columns = [
        "Cluster",
        "Late Delivery Rate"
    ]

    late_rate["Cluster"] = (
        "Cluster " +
        late_rate["Cluster"].astype(str)
    )

    fig_late = px.bar(
        late_rate,
        x="Cluster",
        y="Late Delivery Rate",
        text="Late Delivery Rate",
        title="Share of Late Deliveries by Cluster",
        labels={
            "Late Delivery Rate": "Late Deliveries (%)"
        }
    )

    fig_late.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_late.update_layout(
        height=450,
        yaxis=dict(
            range=[0, 100]
        )
    )

    st.plotly_chart(
        fig_late,
        use_container_width=True
    )

# ---------------- Association Rules ----------------
elif page == "Association Rules":
    st.subheader("Association Rules: What tends to occur together?")

    s1, s2 = st.columns(2)
    with s1:
        min_support = st.slider("Minimum support", 0.01, 0.15, 0.03, 0.01)
    with s2:
        min_confidence = st.slider("Minimum confidence", 0.10, 0.90, 0.30, 0.05)

    rules = association_rules(filtered, min_support, min_confidence)

    if rules.empty:
        st.warning("No rules satisfy the current thresholds. Lower support/confidence.")
    else:
        display = rules.copy()
        display["support"] = (display["support"] * 100).round(2)
        display["confidence"] = (display["confidence"] * 100).round(2)
        display["lift"] = display["lift"].round(3)

        st.dataframe(
            display.rename(columns={
                "support": "support (%)",
                "confidence": "confidence (%)"
            }),
            use_container_width=True,
            hide_index=True
        )

        top = rules.head(15).copy()
        top["rule"] = top["antecedent"] + " → " + top["consequent"]

        fig = px.bar(
            top.sort_values("lift"),
            x="lift", y="rule", orientation="h",
            title="Strongest association rules by lift",
            labels={"lift": "Lift"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Lift > 1 means the consequent occurs more often when the antecedent "
            "is present than would be expected from the consequent's overall frequency."
        )

# ---------------- Supply Map ----------------
elif page == "Supply Map":
    import json

    st.subheader("Where is it made and where does it go?")

    # --- manufacturing site -> country, from the supplied JSON mapping ---
    with open("manufacturing_site.json", encoding="utf-8") as f:
        SITE_TO_COUNTRY = json.load(f)

    # --- country name -> ISO-3 code for the map ---
    ISO3 = {
        "Afghanistan": "AFG", "Angola": "AGO", "Australia": "AUS", "Belize": "BLZ",
        "Benin": "BEN", "Botswana": "BWA", "Burkina Faso": "BFA", "Burundi": "BDI",
        "Cameroon": "CMR", "Canada": "CAN", "China": "CHN", "Congo, DRC": "COD",
        "Côte d'Ivoire": "CIV", "Cyprus": "CYP", "Dominican Republic": "DOM",
        "Ethiopia": "ETH", "France": "FRA", "Germany": "DEU", "Ghana": "GHA",
        "Greece": "GRC", "Guatemala": "GTM", "Guinea": "GIN", "Guyana": "GUY",
        "Haiti": "HTI", "India": "IND", "Ireland": "IRL", "Israel": "ISR",
        "Italy": "ITA", "Japan": "JPN", "Kazakhstan": "KAZ", "Kenya": "KEN",
        "Kyrgyzstan": "KGZ", "Lebanon": "LBN", "Lesotho": "LSO", "Liberia": "LBR",
        "Libya": "LBY", "Malawi": "MWI", "Mali": "MLI", "Mozambique": "MOZ",
        "Namibia": "NAM", "Netherlands": "NLD", "Nigeria": "NGA", "Norway": "NOR",
        "Pakistan": "PAK", "Poland": "POL", "Puerto Rico": "PRI", "Rwanda": "RWA",
        "Senegal": "SEN", "Sierra Leone": "SLE", "South Africa": "ZAF",
        "South Korea": "KOR", "South Sudan": "SSD", "Spain": "ESP", "Sudan": "SDN",
        "Swaziland": "SWZ", "Switzerland": "CHE", "Tanzania": "TZA",
        "Thailand": "THA", "Togo": "TGO", "Uganda": "UGA",
        "United Kingdom": "GBR", "United States of America": "USA",
        "Vietnam": "VNM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
    }

    m1, m2 = st.columns(2)

    with m1:
        metric_label = st.radio(
            "Map metric",
            ["Shipments", "Line-item value (USD)", "Freight cost (USD)"],
            horizontal=True,
        )

    with m2:
        map_view = st.radio(
                "Map view",
                ["Manufacturing", "Destination", "Both"],
                horizontal=True,
            )

    log_scale = st.checkbox(
        "Log colour scale (recommended: stops large outliers washing out the map)",
        value=True,
        )
    
    geo = filtered.copy()
    geo["manufacturing country"] = (
        geo["manufacturing site"].map(SITE_TO_COUNTRY).fillna("Unknown")
    )

    def country_summary(data, col):
        g = data.groupby(col).agg(
            shipments=("id", "count"),
            value=("line item value", "sum"),
            freight=("freight cost (usd)", "sum"),
        ).reset_index().rename(columns={col: "country"})
        g["iso3"] = g["country"].map(ISO3)
        return g

    metric_col = {
        "Shipments": "shipments",
        "Line-item value (USD)": "value",
        "Freight cost (USD)": "freight",
    }[metric_label]

    mfg = country_summary(geo[geo["manufacturing country"] != "Unknown"], "manufacturing country")
    dest = country_summary(geo.dropna(subset=["country"]), "country")

    m1, m2, m3 = st.columns(3)
    m1.metric("Manufacturing countries", f"{mfg['country'].nunique():,}")
    m2.metric("Delivery countries", f"{dest['country'].nunique():,}")
    unknown_pct = 100 * (geo["manufacturing country"] == "Unknown").mean() if len(geo) else 0
    m3.metric("Shipments with unknown origin", f"{unknown_pct:.1f}%")


    # ============================================================
    # SINGLE MAP WITH VIEW TOGGLE
    # ============================================================

    def draw_single_map(view, scale_mfg="Blues", scale_dest="Oranges"):

        fig = go.Figure()

        # ========================================================
        # MANUFACTURING
        # ========================================================

        if view == "Manufacturing":

            plot_df = mfg.dropna(subset=["iso3"]).copy()

            if log_scale:
                plot_df["color_val"] = np.log10(
                    plot_df[metric_col].clip(lower=1)
                )
            else:
                plot_df["color_val"] = plot_df[metric_col]

            fig.add_trace(
                go.Choropleth(
                    locations=plot_df["iso3"],
                    z=plot_df["color_val"],
                    text=plot_df["country"],

                    customdata=np.stack([
                        plot_df["shipments"],
                        plot_df["value"],
                        plot_df["freight"],
                    ], axis=-1),

                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Manufacturing<br>"
                        "Shipments: %{customdata[0]:,.0f}<br>"
                        "Line-item value: $%{customdata[1]:,.0f}<br>"
                        "Freight cost: $%{customdata[2]:,.0f}"
                        "<extra></extra>"
                    ),

                    colorscale=scale_mfg,
                    colorbar=dict(title=metric_label),

                    marker_line_color="white",
                    marker_line_width=0.5,

                    name="Manufacturing",
                )
            )

        # ========================================================
        # DESTINATION
        # ========================================================

        elif view == "Destination":

            plot_df = dest.dropna(subset=["iso3"]).copy()

            if log_scale:
                plot_df["color_val"] = np.log10(
                    plot_df[metric_col].clip(lower=1)
                )
            else:
                plot_df["color_val"] = plot_df[metric_col]

            fig.add_trace(
                go.Choropleth(
                    locations=plot_df["iso3"],
                    z=plot_df["color_val"],
                    text=plot_df["country"],

                    customdata=np.stack([
                        plot_df["shipments"],
                        plot_df["value"],
                        plot_df["freight"],
                    ], axis=-1),

                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Destination<br>"
                        "Shipments: %{customdata[0]:,.0f}<br>"
                        "Line-item value: $%{customdata[1]:,.0f}<br>"
                        "Freight cost: $%{customdata[2]:,.0f}"
                        "<extra></extra>"
                    ),

                    colorscale=scale_dest,
                    colorbar=dict(title=metric_label),

                    marker_line_color="white",
                    marker_line_width=0.5,

                    name="Destination",
                )
            )

        # ========================================================
        # BOTH
        # ========================================================

        else:

            # ----------------------------------------------------
            # DESTINATION ONLY
            # ----------------------------------------------------

            dest_only = dest[
                ~dest["country"].isin(mfg["country"])
            ].dropna(subset=["iso3"]).copy()

            if len(dest_only):

                if log_scale:
                    dest_only["color_val"] = np.log10(
                        dest_only[metric_col].clip(lower=1)
                    )
                else:
                    dest_only["color_val"] = dest_only[metric_col]

                fig.add_trace(
                    go.Choropleth(
                        locations=dest_only["iso3"],
                        z=dest_only["color_val"],
                        text=dest_only["country"],

                        customdata=np.stack([
                            dest_only["shipments"],
                            dest_only["value"],
                            dest_only["freight"],
                        ], axis=-1),

                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Destination only<br>"
                            "Shipments: %{customdata[0]:,.0f}<br>"
                            "Line-item value: $%{customdata[1]:,.0f}<br>"
                            "Freight cost: $%{customdata[2]:,.0f}"
                            "<extra></extra>"
                        ),

                        colorscale=scale_dest,

                        colorbar=dict(
                            title="Destination",
                            x=1.02,
                            y=0.25,
                            len=0.4,
                        ),

                        marker_line_color="white",
                        marker_line_width=0.5,

                        name="Destination",
                    )
                )

            # ----------------------------------------------------
            # MANUFACTURING ONLY
            # ----------------------------------------------------

            mfg_only = mfg[
                ~mfg["country"].isin(dest["country"])
            ].dropna(subset=["iso3"]).copy()

            if len(mfg_only):

                if log_scale:
                    mfg_only["color_val"] = np.log10(
                        mfg_only[metric_col].clip(lower=1)
                    )
                else:
                    mfg_only["color_val"] = mfg_only[metric_col]

                fig.add_trace(
                    go.Choropleth(
                        locations=mfg_only["iso3"],
                        z=mfg_only["color_val"],
                        text=mfg_only["country"],

                        customdata=np.stack([
                            mfg_only["shipments"],
                            mfg_only["value"],
                            mfg_only["freight"],
                        ], axis=-1),

                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Manufacturing only<br>"
                            "Shipments: %{customdata[0]:,.0f}<br>"
                            "Line-item value: $%{customdata[1]:,.0f}<br>"
                            "Freight cost: $%{customdata[2]:,.0f}"
                            "<extra></extra>"
                        ),

                        colorscale=scale_mfg,
                        showscale=False,

                        marker_line_color="white",
                        marker_line_width=0.5,

                        name="Manufacturing",
                    )
                )

            # ----------------------------------------------------
            # BOTH MANUFACTURING + DESTINATION
            # ----------------------------------------------------

            overlap = (
                mfg[
                    mfg["country"].isin(dest["country"])
                ]
                .merge(
                    dest,
                    on=["country", "iso3"],
                    suffixes=("_mfg", "_dest"),
                )
            )

            if len(overlap):

                if log_scale:
                    overlap["color_val"] = np.log10(
                        overlap[f"{metric_col}_mfg"].clip(lower=1)
                    )
                else:
                    overlap["color_val"] = (
                        overlap[f"{metric_col}_mfg"]
                    )

                fig.add_trace(
                    go.Choropleth(
                        locations=overlap["iso3"],
                        z=overlap["color_val"],
                        text=overlap["country"],

                        customdata=np.stack([
                            overlap["shipments_mfg"],
                            overlap["value_mfg"],
                            overlap["freight_mfg"],
                            overlap["shipments_dest"],
                            overlap["value_dest"],
                            overlap["freight_dest"],
                        ], axis=-1),

                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "<b>Manufacturing + Destination</b>"
                            "<br><br>"
                            "<b>Manufacturing</b><br>"
                            "Shipments: %{customdata[0]:,.0f}<br>"
                            "Line-item value: $%{customdata[1]:,.0f}<br>"
                            "Freight cost: $%{customdata[2]:,.0f}"
                            "<br><br>"
                            "<b>Destination</b><br>"
                            "Shipments: %{customdata[3]:,.0f}<br>"
                            "Line-item value: $%{customdata[4]:,.0f}<br>"
                            "Freight cost: $%{customdata[5]:,.0f}"
                            "<extra></extra>"
                        ),

                        colorscale=scale_mfg,
                        showscale=True,

                        colorbar=dict(
                            title="Manufacturing",
                            x=1.02,
                            y=0.75,
                            len=0.4,
                        ),

                        marker_line_color="#ff4d4d",
                        marker_line_width=3,

                        name="Both",
                    )
                )

        # ========================================================
        # ORIGINAL 2D MAP
        # ========================================================

        fig.update_geos(
            showcountries=True,
            showframe=False,
            projection_type="natural earth",
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            height=520,
        )

        return fig


    # ============================================================
    # DISPLAY ONE MAP
    # ============================================================

    st.plotly_chart(
        draw_single_map(map_view),
        use_container_width=True,
    )

    st.markdown("#### Country rankings")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Top manufacturing countries**")
        st.dataframe(
            mfg.sort_values(metric_col, ascending=False)
            .drop(columns="iso3").head(15).round(0),
            use_container_width=True, hide_index=True,
        )
    with t2:
        st.markdown("**Top delivery countries**")
        st.dataframe(
            dest.sort_values(metric_col, ascending=False)
            .drop(columns="iso3").head(15).round(0),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Manufacturing country is taken from the **manufacturing_site.json** mapping.  \n"
        "Shipments with no identifiable site ('Not Applicable') are shown as Unknown "
        "and left off the map."
    )

# ---------------- Insights ----------------
elif page == "Insights & Hypothests Testing":
    # ============================================================
    # HYPOTHESIS TEST: FREIGHT COST vs DELIVERY DELAY
    # ============================================================

    st.subheader("📊 Does Freight Cost Affect Delivery Delay?")

    st.markdown(
        """
        **H₀:** Freight cost and delivery delay are independent.  
        **H₁:** Freight cost and delivery delay are statistically dependent.
        
        A Chi-square test of independence is used at a significance level
        of **α = 0.05**.
        """
    )

    # ------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------

    test_data = filtered[
        ["freight cost (usd)", "delivery_delay_days"]
    ].copy()

    test_data["freight cost (usd)"] = pd.to_numeric(
        test_data["freight cost (usd)"],
        errors="coerce"
    )

    test_data["delivery_delay_days"] = pd.to_numeric(
        test_data["delivery_delay_days"],
        errors="coerce"
    )

    test_data = test_data.dropna()


    # ------------------------------------------------------------
    # Categorize freight cost
    # ------------------------------------------------------------

    test_data["freight_category"] = pd.qcut(
        test_data["freight cost (usd)"],
        q=4,
        labels=[
            "Low",
            "Medium-Low",
            "Medium-High",
            "High"
        ],
        duplicates="drop"
    )


    # ------------------------------------------------------------
    # Categorize delivery delay
    # ------------------------------------------------------------

    test_data["delay_category"] = pd.cut(
        test_data["delivery_delay_days"],
        bins=[
            -float("inf"),
            0,
            7,
            30,
            float("inf")
        ],
        labels=[
            "Early/On Time",
            "1-7 days late",
            "8-30 days late",
            ">30 days late"
        ]
    )


    # ------------------------------------------------------------
    # Contingency table
    # ------------------------------------------------------------

    contingency_table = pd.crosstab(
        test_data["freight_category"],
        test_data["delay_category"]
    )


    # ------------------------------------------------------------
    # Chi-square test
    # ------------------------------------------------------------

    if (
        contingency_table.shape[0] >= 2
        and contingency_table.shape[1] >= 2
    ):

        chi2, p, dof, expected = chi2_contingency(
            contingency_table
        )

        alpha = 0.05

        if p < alpha:
            conclusion = (
                "Reject H₀ — freight cost and delivery delay "
                "are statistically dependent."
            )
            conclusion_type = "success"
        else:
            conclusion = (
                "Fail to reject H₀ — there is insufficient evidence "
                "to conclude that freight cost and delivery delay "
                "are dependent."
            )
            conclusion_type = "info"


        # ========================================================
        # RESULTS
        # ========================================================

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Chi-square statistic",
            f"{chi2:.3f}"
        )

        r2.metric(
            "Degrees of freedom",
            f"{dof}"
        )

        r3.metric(
            "p-value",
            f"{p:.4g}"
        )

        r4.metric(
            "Significance level",
            "0.05"
        )


        if conclusion_type == "success":
            st.error(conclusion)
        else:
            st.success(conclusion)


        # ========================================================
        # CONTINGENCY TABLE + HEATMAP
        # ========================================================

        st.markdown("#### Freight Cost vs Delivery Delay")

        st.dataframe(
            contingency_table,
            use_container_width=True
        )


        # ========================================================
        # INTERPRETATION
        # ========================================================

        st.markdown("#### Interpretation")

        if p < alpha:
            st.markdown(
                f"""
                The p-value (**{p:.4g}**) is below 0.05. Therefore, we
                **reject the null hypothesis**. The data provides statistical
                evidence of an association between freight-cost category and
                delivery-delay category.
                
                This indicates that delivery delay is **not independent of
                freight cost** in the dataset.
                """
            )

        else:
            st.markdown(
                f"""
                The p-value (**{p:.4g}**) is above 0.05. Therefore, we
                **fail to reject the null hypothesis**. There is insufficient
                statistical evidence of an association between freight-cost
                category and delivery-delay category.
                
                This does not prove that the variables are independent; it
                means that this test did not find sufficient evidence of
                dependence at the 5% significance level.
                """
            )


        # ========================================================
        # EXPECTED FREQUENCIES CHECK
        # ========================================================

        expected_df = pd.DataFrame(
            expected,
            index=contingency_table.index,
            columns=contingency_table.columns
        )

        low_expected = (expected < 5).sum()

        if low_expected > 0:

            st.warning(
                f"""
                **Chi-square assumption warning:** {low_expected} cells have
                expected frequencies below 5. The chi-square approximation
                may therefore be unreliable.
                """
            )

        else:

            st.caption(
                "All expected cell frequencies are ≥ 5, so the standard "
                "chi-square approximation is appropriate."
            )

    else:

        st.warning(
            "Not enough categorical data is available to perform the "
            "chi-square test."
        )

    st.divider()

    st.subheader("📊 Delivery Delay: Before 2010 vs 2010–2014")

    # Prepare data
    before = filtered.loc[
        filtered["delivery_year"] < 2010,
        "delivery_delay_days"
    ].dropna()

    period = filtered.loc[
        filtered["delivery_year"].between(2010, 2014),
        "delivery_delay_days"
    ].dropna()

    # Welch's independent t-test
    t_stat, p = ttest_ind(
        before,
        period,
        equal_var=False
    )

    # Calculate means
    before_mean = before.mean()
    period_mean = period.mean()

    st.write(
        "**H₀:** There is no significant difference in mean delivery delays "
        "between the two periods."
    )

    st.write(
        "**H₁:** There is a significant difference in mean delivery delays "
        "between the two periods."
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Before 2010 Mean Delay",
            f"{before_mean:.2f} days"
        )

    with col2:
        st.metric(
            "2010–2014 Mean Delay",
            f"{period_mean:.2f} days"
        )

    with col3:
        st.metric(
            "t-statistic",
            f"{t_stat:.4f}"
        )

    with col4:
        st.metric(
            "p-value",
            f"{p:.6f}"
        )

    # Verdict
    if p < 0.05:
        st.error(
            "Reject H₀: There is a statistically significant difference "
            "in delivery delays between the two periods."
        )
    else:
        st.success(
            "Fail to reject H₀: There is no statistically significant "
            "difference in delivery delays between the two periods."
        )

    comparison = pd.DataFrame({
        "Period": ["Before 2010", "2010–2014"],
        "Mean Delay": [before_mean, period_mean]
    })

    difference = period_mean - before_mean

    if difference > 0:
        st.write(
            f"Delivery delays increased by approximately **{difference:.2f} days** "
            f"during 2010–2014 compared with the period before 2010."
        )
    else:
        st.write(
            f"Delivery delays decreased by approximately **{abs(difference):.2f} days** "
            f"during 2010–2014 compared with the period before 2010."
        )

    st.divider()




    st.markdown(
        """
        ### 1. Look beyond total freight cost
        A country can have a high total freight bill simply because it receives many
        shipments. **Freight per kg** and **freight per unit** help distinguish volume
        effects from logistics efficiency.

        ### 2. Treat missingness as information
        Missing shipment mode, insurance, and process dates should not automatically
        be replaced with zero. The dashboard keeps them explicit so that the analysis
        does not silently invent operational facts.

        ### 3. Use PCA to explain clustering
        PCA is not being used just as a visual decoration. Its loadings identify the
        combinations of cost, value, quantity and weight that dominate the data before
        K-Means groups shipments.

        ### 4. Investigate high-cost outliers
        Shipments with unusually high freight-per-unit or freight-per-kg are candidates
        for deeper operational review. They are **flags**, not proof of inefficiency.

        ### 5. Association rules describe recurring operating patterns
        High-lift rules can reveal combinations of product group, shipment mode,
        fulfilment method and Incoterm that repeatedly appear together.
        """
    )

    st.markdown("#### Automatically generated watchlist")
    watch = filtered.dropna(subset=["freight_per_unit"]).copy()
    if len(watch):
        threshold = watch["freight_per_unit"].quantile(0.95)
        watch = watch[watch["freight_per_unit"] >= threshold]
        cols = [
            "id", "country", "product group", "shipment mode",
            "line item quantity", "freight cost (usd)",
            "freight_per_unit", "vendor"
        ]
        st.write(
            f"Showing the top 5% by freight per unit "
            f"(threshold ≈ ${threshold:,.2f}/unit)."
        )
        st.dataframe(
            watch.sort_values("freight_per_unit", ascending=False)[cols].head(25),
            use_container_width=True,
            hide_index=True
        )

st.divider()
st.caption(
    "Academic analytics dashboard • Source: supplied Supply Chain Shipment dataset • "
    "All derived metrics are calculated from the filtered data."
)
