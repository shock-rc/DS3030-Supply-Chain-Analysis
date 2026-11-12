from .common import (
    go,
    np,
    pd,
    px,
    st,
    StandardScaler,
    KMeans,
    PCA,
)

from kmodes.kprototypes import KPrototypes


def render(filtered: pd.DataFrame, df: pd.DataFrame) -> None:

    # ============================================================
    # FEATURES
    # ============================================================

    numeric_features = [
        "line item quantity",
        "line item value",
        "unit price",
        "weight (kilograms)",
        "freight cost (usd)",
        "line item insurance (usd)",
        "freight_per_kg",
        "freight_pct_of_value",
        "delivery_delay_days",
    ]

    categorical_features = [
        "country",
        "shipment mode",
        "product group",
        "vendor",
        "vendor inco term",
        "fulfill via",
        "delivery_status",
    ]

    with st.expander("Clustering Features", expanded=True):

        algorithm = st.radio(
            "Clustering algorithm",
            ["K-Means", "K-Prototypes"],
            horizontal=True,
            index=1,
        )

        num_cols = st.multiselect(
            "Numeric features",
            numeric_features,
            default=numeric_features,
        )

        cat_cols = st.multiselect(
            "Categorical features",
            categorical_features,
            default= categorical_features,
            disabled=(algorithm == "K-Means"),
        )

    if not num_cols:
        st.warning("Select at least one numeric feature.")
        return

    if algorithm == "K-Prototypes" and not cat_cols:
        st.warning("Select at least one categorical feature.")
        return

    # ============================================================
    # PREPARE DATA
    # ============================================================

    d = filtered.copy()

    d = d[
        (d["weight (kilograms)"] > 0) &
        (d["line item value"] > 0)
    ].copy()

    if d.empty:
        st.warning("No valid observations are available.")
        return

    # Log-transform numeric variables
    numeric_data = pd.DataFrame(index=d.index)

    for col in num_cols:
        values = pd.to_numeric(d[col], errors="coerce")

        numeric_data[col] = (
            np.sign(values) * np.log1p(values.abs())
        )

    numeric_data = (
        numeric_data
        .replace([np.inf, -np.inf], np.nan)
        .fillna(numeric_data.median())
    )

    # Remove constant features
    numeric_data = numeric_data.loc[
        :,
        numeric_data.std() > 1e-8
    ]

    if numeric_data.empty:
        st.warning("No usable numeric features remain.")
        return

    # Standardize numeric features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(numeric_data)

    # ============================================================
    # CLUSTERING
    # ============================================================

    K = st.slider(
        "Number of clusters (k)",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
    )

    if algorithm == "K-Means":

        model = KMeans(
            n_clusters=K,
            n_init=10,
            random_state=42,
        )

        labels = model.fit_predict(Xs)

    else:

        categorical_data = (
            d[cat_cols]
            .fillna("Unknown")
            .astype(str)
        )

        model_input = np.column_stack([
            Xs,
            categorical_data.to_numpy(),
        ])

        categorical_indices = list(
            range(
                Xs.shape[1],
                Xs.shape[1] + len(cat_cols),
            )
        )

        model = KPrototypes(
            n_clusters=K,
            init="Cao",
            n_init=5,
            max_iter=100,
            gamma=None,
            random_state=42,
        )

        labels = model.fit_predict(
            model_input,
            categorical=categorical_indices,
        )

    d["cluster"] = labels

    # ============================================================
    # CLUSTER SIZES
    # ============================================================

    size = (
        d["cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_size_df = pd.DataFrame({
        "Cluster": [
            f"Cluster {c}" for c in size.index
        ],
        "Observations": size.values,
        "Percentage": (
            size.values / len(d) * 100
        ).round(1),
    })

    fig_size = px.bar(
        cluster_size_df,
        x="Cluster",
        y="Observations",
        text="Observations",
        title=f"{algorithm}: Cluster Sizes",
    )

    fig_size.update_traces(
        textposition="outside"
    )

    fig_size.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_size,
        use_container_width=True,
    )

    # ============================================================
    # PCA VISUALIZATION
    # ============================================================

    st.subheader(f"{algorithm} Clusters in PCA Space")

    if Xs.shape[1] >= 2:

        pca = PCA(n_components=2)
        P = pca.fit_transform(Xs)
        variance = (
            pca.explained_variance_ratio_ * 100
        )

    else:

        P = np.column_stack([
            Xs[:, 0],
            np.zeros(len(d)),
        ])

        variance = np.array([
            100.0,
            0.0,
        ])

    # Aggregate points for visualization
    nb = 45

    bx = pd.cut(
        P[:, 0],
        nb,
        labels=False,
    )

    by = pd.cut(
        P[:, 1],
        nb,
        labels=False,
    )

    b = pd.DataFrame({
        "x": P[:, 0],
        "y": P[:, 1],
        "bin": bx * nb + by,
        "cluster": labels,
    })

    agg = (
        b.groupby("bin")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            n=("x", "size"),
            cluster=(
                "cluster",
                lambda x: x.mode()[0],
            ),
            purity=(
                "cluster",
                lambda x:
                x.value_counts(
                    normalize=True
                ).iloc[0],
            ),
        )
        .reset_index(drop=True)
    )

    fig_pca = go.Figure()

    for c in sorted(d["cluster"].unique()):

        a = agg[
            agg["cluster"] == c
        ]

        fig_pca.add_trace(
            go.Scatter(
                x=a["x"],
                y=a["y"],
                mode="markers",
                name=f"Cluster {c}",
                marker=dict(
                    size=8,
                    opacity=0.8,
                ),
                customdata=np.column_stack([
                    a["n"],
                    a["purity"],
                ]),
                hovertemplate=(
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "Observations: %{customdata[0]:,.0f}<br>"
                    "Cluster purity: %{customdata[1]:.2%}"
                    "<extra></extra>"
                ),
            )
        )

    # Numeric cluster centers for PCA visualization
    numeric_centers = np.vstack([
        Xs[labels == c].mean(axis=0)
        for c in sorted(d["cluster"].unique())
    ])

    center_pca = (
        pca.transform(numeric_centers)
        if Xs.shape[1] >= 2
        else numeric_centers[:, :1]
    )

    if Xs.shape[1] < 2:
        center_pca = np.column_stack([
            center_pca[:, 0],
            np.zeros(len(center_pca)),
        ])

    fig_pca.add_trace(
        go.Scatter(
            x=center_pca[:, 0],
            y=center_pca[:, 1],
            mode="markers",
            name="Cluster Centers",
            marker=dict(
                symbol="x",
                size=16,
                color="black",
            ),
        )
    )

    fig_pca.update_layout(
        title=f"{algorithm} (k={K}) in PCA Space",
        xaxis_title=(
            f"PC1 ({variance[0]:.1f}% variance)"
        ),
        yaxis_title=(
            f"PC2 ({variance[1]:.1f}% variance)"
        ),
        height=700,
        hovermode="closest",
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True,
    )

    st.caption(
        "PCA is used only for visualization. "
        "K-Prototypes clustering uses both numerical and "
        "categorical features."
    )

    # ============================================================
    # STANDARDIZED CLUSTER PROFILE
    # ============================================================

    st.subheader("Standardized Cluster Profile")

    profile = (
        pd.DataFrame(
            Xs,
            columns=numeric_data.columns,
            index=d.index,
        )
        .groupby(d["cluster"])
        .mean()
    )

    fig_heatmap = px.imshow(
        profile,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.5,
        zmax=1.5,
        color_continuous_midpoint=0,
        labels={
            "x": "Feature",
            "y": "Cluster",
            "color": "Mean z-score",
        },
        title="Cluster Profile",
    )

    fig_heatmap.update_layout(
        height=500,
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True,
    )

    # ============================================================
    # BOXPLOT
    # ============================================================

    st.subheader("Key Variables by Cluster")

    key = [
        "line item value",
        "weight (kilograms)",
        "freight cost (usd)",
        "freight_pct_of_value",
        "freight_per_kg",
        "line item quantity",
    ]

    selected_box_col = st.selectbox(
        "Select variable",
        key,
    )

    box_data = d[
        ["cluster", selected_box_col]
    ].copy()

    box_data[selected_box_col] = (
        pd.to_numeric(
            box_data[selected_box_col],
            errors="coerce",
        )
        .clip(lower=1e-2)
    )

    fig_box = px.box(
        box_data,
        x="cluster",
        y=selected_box_col,
        color="cluster",
        points=False,
        log_y=True,
        title=f"{selected_box_col} by Cluster",
    )

    fig_box.update_layout(
        height=550,
        showlegend=False,
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True,
    )

    # ============================================================
    # CATEGORICAL COMPOSITION
    # ============================================================

    st.subheader("Categorical Composition by Cluster")

    available_cat_plots = [
        col for col in categorical_features
        if col in d.columns
    ]

    selected_cat = st.selectbox(
        "Select categorical variable",
        available_cat_plots,
    )

    cat_pct = (
        pd.crosstab(
            d["cluster"],
            d[selected_cat].astype(str),
            normalize="index",
        )
        .mul(100)
        .reset_index()
        .melt(
            id_vars="cluster",
            var_name=selected_cat,
            value_name="percentage",
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
            "percentage": "% of Cluster",
        },
    )

    fig_cat.update_layout(
        height=600,
        yaxis=dict(range=[0, 100]),
    )

    st.plotly_chart(
        fig_cat,
        use_container_width=True,
    )

    # ============================================================
    # COUNTRY COMPOSITION
    # ============================================================

    st.subheader("Country Composition")

    top_countries = (
        d["country"]
        .value_counts()
        .head(8)
        .index
    )

    country_pct = (
        pd.crosstab(
            d["country"],
            d["cluster"],
            normalize="index",
        )
        .mul(100)
        .loc[top_countries]
        .reset_index()
        .melt(
            id_vars="country",
            var_name="cluster",
            value_name="percentage",
        )
    )

    country_pct["cluster"] = (
        "Cluster "
        + country_pct["cluster"].astype(str)
    )

    fig_country = px.bar(
        country_pct,
        x="country",
        y="percentage",
        color="cluster",
        barmode="stack",
        title="Top 8 Countries: Cluster Composition",
        labels={"percentage": "%"},
    )

    fig_country.update_layout(
        height=600,
        yaxis=dict(range=[0, 100]),
        xaxis_tickangle=-40,
    )

    st.plotly_chart(
        fig_country,
        use_container_width=True,
    )

    # ============================================================
    # YEAR COMPOSITION
    # ============================================================

    if "delivery_year" in d.columns:

        year_pct = (
            pd.crosstab(
                d["delivery_year"],
                d["cluster"],
                normalize="index",
            )
            .mul(100)
            .reset_index()
            .melt(
                id_vars="delivery_year",
                var_name="cluster",
                value_name="percentage",
            )
        )

        year_pct["cluster"] = (
            "Cluster "
            + year_pct["cluster"].astype(str)
        )

        fig_year = px.bar(
            year_pct,
            x="delivery_year",
            y="percentage",
            color="cluster",
            barmode="stack",
            title="Delivery Year: Cluster Composition",
            labels={
                "delivery_year": "Delivery Year",
                "percentage": "%",
            },
        )

        fig_year.update_layout(
            height=600,
            yaxis=dict(range=[0, 100]),
        )

        st.plotly_chart(
            fig_year,
            use_container_width=True,
        )

    # ============================================================
    # LATE DELIVERY RATE
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
        "Late Delivery Rate",
    ]

    late_rate["Cluster"] = (
        "Cluster "
        + late_rate["Cluster"].astype(str)
    )

    fig_late = px.bar(
        late_rate,
        x="Cluster",
        y="Late Delivery Rate",
        text="Late Delivery Rate",
        title="Share of Late Deliveries by Cluster",
        labels={
            "Late Delivery Rate": "Late Deliveries (%)",
        },
    )

    fig_late.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig_late.update_layout(
        height=450,
        yaxis=dict(range=[0, 100]),
    )

    st.plotly_chart(
        fig_late,
        use_container_width=True,
    )