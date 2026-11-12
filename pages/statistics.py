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
import numpy as np

def render(filtered: pd.DataFrame, df: pd.DataFrame) -> None:
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

    st.markdown("#### Scaled Plots")
    cols = [
        "line item quantity",
        "line item value",
        "weight (kilograms)",
        "freight cost (usd)"
    ]
    a, b = st.columns(2)
    for i, c in enumerate(cols):

        hist = filtered.dropna(subset=[c])
        # Same transformation as:
        # np.log10(df[c].dropna().clip(lower=1))
        hist["log10_value"] = np.log10(hist[c].clip(lower=1))
        fig = px.histogram(
            hist,
            x="log10_value",
            nbins=40,
            title=f"{c} (log10 scale)",
            labels={
                "log10_value": "log10(value)"
            }
        )

        fig.update_layout(
            xaxis_title="log10(value)",
            yaxis_title="Rows"
        )

        if i % 2 == 0:
            with a:
                st.plotly_chart(fig, use_container_width=True)
        else:
            with b:
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Categorical plots")
    a, b = st.columns(2)

    with a:
        product_counts = (
            filtered["product group"]
            .value_counts()
            .reset_index()
        )

        product_counts.columns = ["product group", "count"]

        fig = px.bar(
            product_counts,
            x="product group",
            y="count",
            title="Product group",
            labels={
                "product_group": "",
                "count": "Rows"
            }
        )

        st.plotly_chart(fig, use_container_width=True)


    with b:
        fulfill_counts = (
            filtered["fulfill via"]
            .value_counts()
            .reset_index()
        )

        fulfill_counts.columns = ["fulfill via", "count"]

        fig = px.bar(
            fulfill_counts,
            x="fulfill via",
            y="count",
            title="Fulfill via",
            labels={
                "fulfill via": "",
                "count": "Rows"
            }
        )

        st.plotly_chart(fig, use_container_width=True)


    a, b = st.columns(2)

    with a:
        shipment_counts = (
            filtered["shipment mode"]
            .value_counts(dropna=False)
            .reset_index()
        )

        shipment_counts.columns = ["shipment mode", "count"]

        # Make NaN explicitly visible
        shipment_counts["shipment mode"] = (
            shipment_counts["shipment mode"]
            .fillna("NaN")
        )

        fig = px.bar(
            shipment_counts,
            x="shipment mode",
            y="count",
            title="Shipment mode (NaN included)",
            labels={
                "shipment mode": "",
                "count": "Rows"
            }
        )

        st.plotly_chart(fig, use_container_width=True)


    with b:
        dosage_counts = (
            filtered["dosage form"]
            .value_counts()
            .head(8)
            .reset_index()
        )

        dosage_counts.columns = ["dosage form", "count"]

        fig = px.bar(
            dosage_counts,
            x="count",
            y="dosage form",
            orientation="h",
            title="Top 8 dosage forms",
            labels={
                "dosage form": "",
                "count": "Rows"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Delay Plots")
    
    delay = filtered["delivery_delay_days"].copy()
    

    late = delay > 0
    mode = filtered["shipment mode"].fillna("Unknown")
    year = filtered["scheduled delivery date"].map(lambda x: int(x[:4]))

    a, b = st.columns(2)

    with a:

        delay_plot = delay.clip(-60, 60).dropna()

        fig = px.histogram(
            x=delay_plot,
            nbins=60,
            title="Delivery delay in days (clipped to ±60)",
            labels={
                "x": "Delivery delay (days)",
                "y": "Rows"
            }
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="red"
        )

        fig.add_annotation(
            x=0,
            y=1,
            yref="paper",
            text="0 = on time",
            showarrow=False,
            yshift=10
        )

        st.plotly_chart(fig, use_container_width=True)

    with b:

        late_by_route = (
            late.groupby(df["fulfill via"])
            .mean()
            .mul(100)
            .reset_index()
        )

        late_by_route.columns = [
            "fulfill via",
            "percent_late"
        ]

        fig = px.bar(
            late_by_route,
            x="fulfill via",
            y="percent_late",
            title="% late by fulfill via",
            labels={
                "fulfill via": "",
                "percent_late": "% late"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)

    with a:

        late_by_mode = (
            late.groupby(mode)
            .mean()
            .mul(100)
            .sort_values()
            .reset_index()
        )

        late_by_mode.columns = [
            "shipment mode",
            "percent_late"
        ]

        fig = px.bar(
            late_by_mode,
            x="shipment mode",
            y="percent_late",
            title="% late by shipment_mode",
            labels={
                "shipment mode": "",
                "percent_late": "% late"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with b:

        late_by_year = (
            late.groupby(year)
            .mean()
            .mul(100)
            .reset_index()
        )

        late_by_year.columns = [
            "year",
            "percent_late"
        ]

        fig = px.line(
            late_by_year,
            x="year",
            y="percent_late",
            markers=True,
            title="% late by scheduled year",
            labels={
                "year": "",
                "percent_late": "% late"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    t = (
        pd.DataFrame({
            "fulfill via": df["fulfill via"],
            "shipment mode": mode,
            "late": late
        })
        .groupby(["fulfill via", "shipment mode"])
        .agg(
            percent_late=("late", "mean"),
            rows=("late", "size")
        )
        .reset_index()
    )

    # Keep only groups with at least 30 rows
    t = t[t["rows"] >= 30].copy()

    # Convert proportion to percentage
    t["percent_late"] *= 100


    fig = px.bar(
        t,
        x="fulfill via",
        y="percent_late",
        color="shipment mode",
        barmode="group",
        title="% late by shipment mode within each route (groups with 30+ rows)",
        labels={
            "fulfill via": "",
            "percent_late": "% late",
            "shipment mode": "Shipment mode"
        },
        hover_data={
            "rows": True,
            "percent_late": ":.1f"
        }
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
