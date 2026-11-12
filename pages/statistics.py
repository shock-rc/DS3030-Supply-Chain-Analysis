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
