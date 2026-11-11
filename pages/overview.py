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
