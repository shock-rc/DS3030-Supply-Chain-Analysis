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
