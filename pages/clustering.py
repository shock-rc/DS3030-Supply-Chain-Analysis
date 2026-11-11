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
    StandardScaler,
    PCA,
    KMeans,
    silhouette_score
)

def render(filtered: pd.DataFrame, df: pd.DataFrame) -> None:
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
