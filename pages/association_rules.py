from .common import pd, px, st
from data import association_rules


def render(filtered: pd.DataFrame, df: pd.DataFrame) -> None:
    st.subheader("Association Rules: What tends to occur together?")

    numeric_data = filtered.select_dtypes(include="number").drop(columns = "id", errors="ignore")

    corr_matrix = numeric_data.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix"
    )

    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Variables",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

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
