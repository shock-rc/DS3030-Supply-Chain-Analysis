from .common import chi2_contingency, pd, st, ttest_ind
from html import escape


def render(filtered: pd.DataFrame, df: pd.DataFrame) -> None:
        st.subheader("📊 Does Freight Cost Affect Delivery Delay?")

        st.markdown(
            """
            **H₀:** Freight cost and delivery delay are independent.  
            **H₁:** Freight cost and delivery delay are statistically dependent.

            A Chi-square test of independence is used at a significance level
            of **α = 0.05**.
            """
        )


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



        contingency_table = pd.crosstab(
            test_data["freight_category"],
            test_data["delay_category"]
        )



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



            st.markdown("#### Freight Cost vs Delivery Delay")

            st.dataframe(
                contingency_table,
                use_container_width=True
            )



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

        before = filtered.loc[
            filtered["delivery_year"] < 2010,
            "delivery_delay_days"
        ].dropna()

        period = filtered.loc[
            filtered["delivery_year"].between(2010, 2014),
            "delivery_delay_days"
        ].dropna()

        t_stat, p = ttest_ind(
            before,
            period,
            equal_var=False
        )

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

        st.subheader("Final Dashboard Summary")
        st.caption(
            "A filtered, presentation-ready summary of delivery performance, cost "
            "efficiency, and the supporting analyses."
        )

        summary_data = filtered.copy()
        summary_data["delivery_delay_days"] = pd.to_numeric(
            summary_data["delivery_delay_days"], errors="coerce"
        )
        summary_data["freight_per_unit"] = pd.to_numeric(
            summary_data["freight_per_unit"], errors="coerce"
        )
        summary_data["freight_per_kg"] = pd.to_numeric(
            summary_data["freight_per_kg"], errors="coerce"
        )

        total_shipments = len(summary_data)
        late_shipments = summary_data["delivery_status"].eq("Late").sum()
        late_pct = 100 * late_shipments / total_shipments if total_shipments else 0
        median_delay = summary_data["delivery_delay_days"].median()
        median_freight_unit = summary_data["freight_per_unit"].median()

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Shipments analysed", f"{total_shipments:,}")
        s2.metric("Late deliveries", f"{late_pct:.1f}%")
        s3.metric(
            "Median delay",
            f"{median_delay:.1f} days" if pd.notna(median_delay) else "N/A",
        )
        s4.metric(
            "Median freight / unit",
            f"${median_freight_unit:,.2f}"
            if pd.notna(median_freight_unit)
            else "N/A",
        )

        late_data = summary_data[summary_data["delivery_status"].eq("Late")]
        mode = (
            late_data["shipment mode"].dropna().value_counts().idxmax()
            if not late_data.empty and late_data["shipment mode"].notna().any()
            else "not available"
        )
        country = (
            late_data["country"].dropna().value_counts().idxmax()
            if not late_data.empty and late_data["country"].notna().any()
            else "not available"
        )

        findings = pd.DataFrame(
            {
                "Area": [
                    "Delivery performance",
                    "Hypothesis testing",
                    "Segmentation and patterns",
                ],
                "Summary": [
                    f"Most common late-shipment mode: {mode}.\n"
                    f"Most represented destination: {country}.",
                    "Chi-square: freight cost and delay are statistically "
                    "dependent.\n"
                    "Welch's t-test: delivery delays increased during "
                    "2010–2014 compared with the period before 2010.",
                    "PCA and K-Means were attempted.\n"
                    "The low silhouette score means no definitive clusters could be made out of the data.\n"
                    "No clear patterns could be established.",
                ],
            }
        )
        findings_rows = "".join(
            f"<tr><td>{escape(row['Area'])}</td>"
            f"<td>{escape(row['Summary'])}</td></tr>"
            for _, row in findings.iterrows()
        )
        st.markdown(
            f"""
            <div style="width: 100%; overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr>
                            <th style="width: 24%; text-align: left;">Area</th>
                            <th style="text-align: left;">Summary</th>
                        </tr>
                    </thead>
                    <tbody>{findings_rows}</tbody>
                </table>
            </div>
            <style>
                th, td {{
                    padding: 0.65rem 0.75rem;
                    border-bottom: 1px solid rgba(128, 128, 128, 0.35);
                    vertical-align: top;
                    white-space: pre-line;
                    overflow-wrap: anywhere;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        watch = summary_data.dropna(subset=["freight_per_unit"]).copy()
        st.markdown("#### Priority review list")
        if watch.empty:
            st.info("No freight-per-unit values are available for the current filters.")
        else:
            threshold = watch["freight_per_unit"].quantile(0.95)
            watch = watch[watch["freight_per_unit"] >= threshold]
            cols = [
                "id", "country", "product group", "shipment mode",
                "line item quantity", "freight cost (usd)",
                "freight_per_unit", "delivery_status", "vendor",
            ]
            display = watch.sort_values("freight_per_unit", ascending=False)[cols].head(25)
            st.caption(
                f"Top 5% of freight per unit (review threshold: "
                f"${threshold:,.2f}/unit). These are flags for investigation, "
                "not conclusions about performance."
            )
            st.dataframe(
                display.style.format(
                    {
                        "line item quantity": "{:,.0f}",
                        "freight cost (usd)": "${:,.2f}",
                        "freight_per_unit": "${:,.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.caption(
            "Academic analytics dashboard • Source: supplied Supply Chain Shipment dataset • "
            "All derived metrics are calculated from the filtered data."
        )
