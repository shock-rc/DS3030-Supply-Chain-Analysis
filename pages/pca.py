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
    with st.expander("Features used for PCA", expanded=True):
        # st.multiselect naturally acts as interactive tags with 'x' remove buttons
        pos_cols = st.multiselect(
            label="Features",
            options=pca_features,
            default=pca_features,  # All selected by default like the image
            label_visibility="collapsed" # Hides the redundant label to look clean like a tag container
        )

    # ============================================================
    # FEATURE SELECTION
    # ============================================================

    d = filtered.copy()

    # Remove invalid zero/negative values
    d = d[
        (d["weight (kilograms)"] > 0) &
        (d["line item value"] > 0)
    ].copy()

    X = pd.DataFrame(index=d.index)

    # Log transform positive, heavily skewed variables
    for c in pos_cols:
        X["log_" + c] = np.log1p(d[c])

    # Signed log for delivery delay because it can be negative
    X["signed_log_delay"] = (
        np.sign(d["delivery_delay_days"]) *
        np.log1p(d["delivery_delay_days"].abs())
    )

    # Replace invalid values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Median imputation
    X = X.fillna(X.median())

    # Remove zero-variance columns
    X = X.loc[:, X.std() > 1e-8]

    # ============================================================
    # STANDARDIZATION
    # ============================================================

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ------------------------------------------------------------
    # Explained variance
    # ------------------------------------------------------------

    pca_full = PCA().fit(Xs)

    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)

    components = list(range(1, len(evr) + 1))

    # Plot explained variance
    fig_ev = go.Figure()

    fig_ev.add_trace(
        go.Bar(
            x=components,
            y=evr * 100,
            name="Individual Variance"
        )
    )

    fig_ev.add_trace(
        go.Scatter(
            x=components,
            y=cum * 100,
            mode="lines+markers",
            name="Cumulative Variance",
        )
    )

    fig_ev.add_hline(
        y=80,
        line_dash="dash",
        line_color="gray",
        annotation_text="80%",
        annotation_position="top left",
    )

    fig_ev.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Individual Variance (%)",
        yaxis2=dict(
            title="Cumulative Variance (%)",
            overlaying="y",
            side="right",
            range=[0, 105]
        ),
        height=550,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_ev,
        use_container_width=True
    )


    # ------------------------------------------------------------
    # PCA with 3 components
    # ------------------------------------------------------------

    pca = PCA(n_components=3).fit(Xs)

    P = pca.transform(Xs)

    v = pca.explained_variance_ratio_ * 100


    # ============================================================
    # PCA LOADINGS
    # ============================================================

    st.markdown("### PCA Component Loadings")

    loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=X.columns,
        columns=["PC1", "PC2", "PC3"]
    )

    st.dataframe(
        loadings.round(3),
        use_container_width=True
    )


    # ============================================================
    # CLUSTERING
    # ============================================================

    st.subheader("K-Means Clustering")

    # ------------------------------------------------------------
    # Evaluate k = 2 to 10
    # ------------------------------------------------------------

    ks = range(2, 11)

    inertia = []
    sil = []

    for k in ks:

        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        ).fit(Xs)

        inertia.append(km.inertia_)

        sil.append(
            silhouette_score(
                Xs,
                km.labels_,
                sample_size=5000,
                random_state=42
            )
        )

    best_k = list(ks)[int(np.argmax(sil))]

    st.write(
        f"**Best k according to silhouette score:** {best_k}"
    )


    # ============================================================
    # ELBOW + SILHOUETTE PLOTS
    # ============================================================

    col1, col2 = st.columns(2)


    # ------------------------------------------------------------
    # Elbow curve
    # ------------------------------------------------------------

    fig_elbow = go.Figure()

    fig_elbow.add_trace(
        go.Scatter(
            x=list(ks),
            y=inertia,
            mode="lines+markers",
            name="Inertia"
        )
    )

    fig_elbow.update_layout(
        title="Elbow Curve",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        height=500
    )

    with col1:
        st.plotly_chart(
            fig_elbow,
            use_container_width=True
        )


    # ------------------------------------------------------------
    # Silhouette score
    # ------------------------------------------------------------

    fig_sil = go.Figure()

    fig_sil.add_trace(
        go.Scatter(
            x=list(ks),
            y=sil,
            mode="lines+markers",
            name="Silhouette Score"
        )
    )

    fig_sil.add_vline(
        x=best_k,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Best k = {best_k}",
        annotation_position="top"
    )

    fig_sil.update_layout(
        title="Silhouette Score vs k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        height=500
    )

    with col2:
        st.plotly_chart(
            fig_sil,
            use_container_width=True
        )


    # ============================================================
    # FINAL K-MEANS MODEL
    # ============================================================

    km = KMeans(
        n_clusters=best_k,
        n_init=10,
        random_state=42
    ).fit(Xs)

    labels = km.labels_

    # Project cluster centroids into PCA space
    cent = pca.transform(km.cluster_centers_)


    # ============================================================
    # CLUSTERS IN PCA SPACE
    # ============================================================

    st.markdown("### K-Means Clusters in PCA Space")

    cluster_plot = pd.DataFrame(
        {
            "PC1": P[:, 0],
            "PC2": P[:, 1],
            "Cluster": labels.astype(str),
            "Line Item Value": d["line item value"].values,
            "Weight (kg)": d["weight (kilograms)"].values,
            "Delivery Delay (days)": d["delivery_delay_days"].values
        },
        index=d.index
    )

    fig_clusters = px.scatter(
        cluster_plot,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=[
            "Line Item Value",
            "Weight (kg)",
            "Delivery Delay (days)"
        ],
        labels={
            "PC1": f"PC1 ({v[0]:.1f}% variance)",
            "PC2": f"PC2 ({v[1]:.1f}% variance)"
        },
        title=f"K-Means (k={best_k}) in PCA Space"
    )

    fig_clusters.update_traces(
        marker=dict(
            size=6,
            opacity=0.4
        )
    )

    # Add cluster centroids
    for c in range(best_k):

        fig_clusters.add_trace(
            go.Scatter(
                x=[cent[c, 0]],
                y=[cent[c, 1]],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=18,
                    color="black",
                    line=dict(width=2)
                ),
                name=f"Centroid {c}",
                showlegend=True
            )
        )

    fig_clusters.update_layout(
        height=700
    )

    st.plotly_chart(
        fig_clusters,
        use_container_width=True
    )


    # ============================================================
    # CLUSTER SIZE
    # ============================================================

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_sizes.columns = ["Cluster", "Observations"]

    cluster_sizes["Cluster"] = (
        cluster_sizes["Cluster"]
        .astype(str)
    )

    fig_size = px.bar(
        cluster_sizes,
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
    # CLUSTER PROFILE
    # ============================================================

    st.markdown("### Cluster Profile — Median Values")

    clustered_data = d.copy()
    clustered_data["cluster"] = labels

    profile = (
        clustered_data
        .groupby("cluster")[pos_cols + ["delivery_delay_days"]]
        .median()
        .round(2)
        .T
    )

    st.dataframe(
        profile,
        use_container_width=True
    )


    # ============================================================
    # OPTIONAL: CLUSTER PROFILE HEATMAP
    # ============================================================

    # Calculate standardized feature means per cluster
    profile_z = (
        pd.DataFrame(
            Xs,
            columns=X.columns,
            index=d.index
        )
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
    )

    fig_heatmap = px.imshow(
        profile_z,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Mean Standardized Feature Values by Cluster"
    )

    fig_heatmap.update_layout(
        height=500,
        xaxis_title="Features",
        yaxis_title="Cluster"
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

    st.markdown("### Cluster Analysis")

    st.markdown("""
    - **Cluster 0:** Small, lightweight shipments with low freight costs but high freight cost per kg.
    - **Cluster 1:** Larger, higher-value shipments with relatively efficient transportation costs.
    - **Cluster 2:** Low-value shipments where freight is disproportionately expensive relative to shipment value.
    - **Cluster 3:** Shipments with **exceptionally high delivery delays**, despite only moderately higher freight costs.

    **Overall:** The clusters reveal distinct supply-chain patterns, with **delivery delays being the defining characteristic of Cluster 3**.
    """)