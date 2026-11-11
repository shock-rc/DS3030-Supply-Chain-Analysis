from .common import (
    chi2_contingency,
    gaussian_kde,
    go,
    json,
    np,
    pd,
    px,
    st,
    ttest_ind,
)

def render(filtered: pd.DataFrame, df: pd.DataFrame) -> None:
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
