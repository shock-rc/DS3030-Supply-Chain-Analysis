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
                <strong>Note:</strong> Only NaN and NaT counts are shown here using df.isna().sum()
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
