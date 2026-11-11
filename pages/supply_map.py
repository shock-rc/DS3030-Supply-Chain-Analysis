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
    import json

    st.subheader("Where is it made and where does it go?")

    # --- manufacturing site -> country, from the supplied JSON mapping ---
    with open("manufacturing_site.json", encoding="utf-8") as f:
        SITE_TO_COUNTRY = json.load(f)

    # --- country name -> ISO-3 code for the map ---
    ISO3 = {
        "Afghanistan": "AFG", "Angola": "AGO", "Australia": "AUS", "Belize": "BLZ",
        "Benin": "BEN", "Botswana": "BWA", "Burkina Faso": "BFA", "Burundi": "BDI",
        "Cameroon": "CMR", "Canada": "CAN", "China": "CHN", "Congo, DRC": "COD",
        "Côte d'Ivoire": "CIV", "Cyprus": "CYP", "Dominican Republic": "DOM",
        "Ethiopia": "ETH", "France": "FRA", "Germany": "DEU", "Ghana": "GHA",
        "Greece": "GRC", "Guatemala": "GTM", "Guinea": "GIN", "Guyana": "GUY",
        "Haiti": "HTI", "India": "IND", "Ireland": "IRL", "Israel": "ISR",
        "Italy": "ITA", "Japan": "JPN", "Kazakhstan": "KAZ", "Kenya": "KEN",
        "Kyrgyzstan": "KGZ", "Lebanon": "LBN", "Lesotho": "LSO", "Liberia": "LBR",
        "Libya": "LBY", "Malawi": "MWI", "Mali": "MLI", "Mozambique": "MOZ",
        "Namibia": "NAM", "Netherlands": "NLD", "Nigeria": "NGA", "Norway": "NOR",
        "Pakistan": "PAK", "Poland": "POL", "Puerto Rico": "PRI", "Rwanda": "RWA",
        "Senegal": "SEN", "Sierra Leone": "SLE", "South Africa": "ZAF",
        "South Korea": "KOR", "South Sudan": "SSD", "Spain": "ESP", "Sudan": "SDN",
        "Swaziland": "SWZ", "Switzerland": "CHE", "Tanzania": "TZA",
        "Thailand": "THA", "Togo": "TGO", "Uganda": "UGA",
        "United Kingdom": "GBR", "United States of America": "USA",
        "Vietnam": "VNM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
    }

    m1, m2 = st.columns(2)

    with m1:
        metric_label = st.radio(
            "Map metric",
            ["Shipments", "Line-item value (USD)", "Freight cost (USD)"],
            horizontal=True,
        )

    with m2:
        map_view = st.radio(
                "Map view",
                ["Manufacturing", "Destination", "Both"],
                horizontal=True,
            )

    log_scale = st.checkbox(
        "Log colour scale (recommended: stops large outliers washing out the map)",
        value=True,
        )

    geo = filtered.copy()
    geo["manufacturing country"] = (
        geo["manufacturing site"].map(SITE_TO_COUNTRY).fillna("Unknown")
    )

    def country_summary(data, col):
        g = data.groupby(col).agg(
            shipments=("id", "count"),
            value=("line item value", "sum"),
            freight=("freight cost (usd)", "sum"),
        ).reset_index().rename(columns={col: "country"})
        g["iso3"] = g["country"].map(ISO3)
        return g

    metric_col = {
        "Shipments": "shipments",
        "Line-item value (USD)": "value",
        "Freight cost (USD)": "freight",
    }[metric_label]

    mfg = country_summary(geo[geo["manufacturing country"] != "Unknown"], "manufacturing country")
    dest = country_summary(geo.dropna(subset=["country"]), "country")

    m1, m2, m3 = st.columns(3)
    m1.metric("Manufacturing countries", f"{mfg['country'].nunique():,}")
    m2.metric("Delivery countries", f"{dest['country'].nunique():,}")
    unknown_pct = 100 * (geo["manufacturing country"] == "Unknown").mean() if len(geo) else 0
    m3.metric("Shipments with unknown origin", f"{unknown_pct:.1f}%")


    # ============================================================
    # SINGLE MAP WITH VIEW TOGGLE
    # ============================================================

    def draw_single_map(view, scale_mfg="Blues", scale_dest="Oranges"):

        fig = go.Figure()

        # ========================================================
        # MANUFACTURING
        # ========================================================

        if view == "Manufacturing":

            plot_df = mfg.dropna(subset=["iso3"]).copy()

            if log_scale:
                plot_df["color_val"] = np.log10(
                    plot_df[metric_col].clip(lower=1)
                )
            else:
                plot_df["color_val"] = plot_df[metric_col]

            fig.add_trace(
                go.Choropleth(
                    locations=plot_df["iso3"],
                    z=plot_df["color_val"],
                    text=plot_df["country"],

                    customdata=np.stack([
                        plot_df["shipments"],
                        plot_df["value"],
                        plot_df["freight"],
                    ], axis=-1),

                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Manufacturing<br>"
                        "Shipments: %{customdata[0]:,.0f}<br>"
                        "Line-item value: $%{customdata[1]:,.0f}<br>"
                        "Freight cost: $%{customdata[2]:,.0f}"
                        "<extra></extra>"
                    ),

                    colorscale=scale_mfg,
                    colorbar=dict(title=metric_label),

                    marker_line_color="white",
                    marker_line_width=0.5,

                    name="Manufacturing",
                )
            )

        # ========================================================
        # DESTINATION
        # ========================================================

        elif view == "Destination":

            plot_df = dest.dropna(subset=["iso3"]).copy()

            if log_scale:
                plot_df["color_val"] = np.log10(
                    plot_df[metric_col].clip(lower=1)
                )
            else:
                plot_df["color_val"] = plot_df[metric_col]

            fig.add_trace(
                go.Choropleth(
                    locations=plot_df["iso3"],
                    z=plot_df["color_val"],
                    text=plot_df["country"],

                    customdata=np.stack([
                        plot_df["shipments"],
                        plot_df["value"],
                        plot_df["freight"],
                    ], axis=-1),

                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Destination<br>"
                        "Shipments: %{customdata[0]:,.0f}<br>"
                        "Line-item value: $%{customdata[1]:,.0f}<br>"
                        "Freight cost: $%{customdata[2]:,.0f}"
                        "<extra></extra>"
                    ),

                    colorscale=scale_dest,
                    colorbar=dict(title=metric_label),

                    marker_line_color="white",
                    marker_line_width=0.5,

                    name="Destination",
                )
            )

        # ========================================================
        # BOTH
        # ========================================================

        else:

            # ----------------------------------------------------
            # DESTINATION ONLY
            # ----------------------------------------------------

            dest_only = dest[
                ~dest["country"].isin(mfg["country"])
            ].dropna(subset=["iso3"]).copy()

            if len(dest_only):

                if log_scale:
                    dest_only["color_val"] = np.log10(
                        dest_only[metric_col].clip(lower=1)
                    )
                else:
                    dest_only["color_val"] = dest_only[metric_col]

                fig.add_trace(
                    go.Choropleth(
                        locations=dest_only["iso3"],
                        z=dest_only["color_val"],
                        text=dest_only["country"],

                        customdata=np.stack([
                            dest_only["shipments"],
                            dest_only["value"],
                            dest_only["freight"],
                        ], axis=-1),

                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Destination only<br>"
                            "Shipments: %{customdata[0]:,.0f}<br>"
                            "Line-item value: $%{customdata[1]:,.0f}<br>"
                            "Freight cost: $%{customdata[2]:,.0f}"
                            "<extra></extra>"
                        ),

                        colorscale=scale_dest,

                        colorbar=dict(
                            title="Destination",
                            x=1.02,
                            y=0.25,
                            len=0.4,
                        ),

                        marker_line_color="white",
                        marker_line_width=0.5,

                        name="Destination",
                    )
                )

            # ----------------------------------------------------
            # MANUFACTURING ONLY
            # ----------------------------------------------------

            mfg_only = mfg[
                ~mfg["country"].isin(dest["country"])
            ].dropna(subset=["iso3"]).copy()

            if len(mfg_only):

                if log_scale:
                    mfg_only["color_val"] = np.log10(
                        mfg_only[metric_col].clip(lower=1)
                    )
                else:
                    mfg_only["color_val"] = mfg_only[metric_col]

                fig.add_trace(
                    go.Choropleth(
                        locations=mfg_only["iso3"],
                        z=mfg_only["color_val"],
                        text=mfg_only["country"],

                        customdata=np.stack([
                            mfg_only["shipments"],
                            mfg_only["value"],
                            mfg_only["freight"],
                        ], axis=-1),

                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Manufacturing only<br>"
                            "Shipments: %{customdata[0]:,.0f}<br>"
                            "Line-item value: $%{customdata[1]:,.0f}<br>"
                            "Freight cost: $%{customdata[2]:,.0f}"
                            "<extra></extra>"
                        ),

                        colorscale=scale_mfg,
                        showscale=False,

                        marker_line_color="white",
                        marker_line_width=0.5,

                        name="Manufacturing",
                    )
                )

            # ----------------------------------------------------
            # BOTH MANUFACTURING + DESTINATION
            # ----------------------------------------------------

            overlap = (
                mfg[
                    mfg["country"].isin(dest["country"])
                ]
                .merge(
                    dest,
                    on=["country", "iso3"],
                    suffixes=("_mfg", "_dest"),
                )
            )

            if len(overlap):

                if log_scale:
                    overlap["color_val"] = np.log10(
                        overlap[f"{metric_col}_mfg"].clip(lower=1)
                    )
                else:
                    overlap["color_val"] = (
                        overlap[f"{metric_col}_mfg"]
                    )

                fig.add_trace(
                    go.Choropleth(
                        locations=overlap["iso3"],
                        z=overlap["color_val"],
                        text=overlap["country"],

                        customdata=np.stack([
                            overlap["shipments_mfg"],
                            overlap["value_mfg"],
                            overlap["freight_mfg"],
                            overlap["shipments_dest"],
                            overlap["value_dest"],
                            overlap["freight_dest"],
                        ], axis=-1),

                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "<b>Manufacturing + Destination</b>"
                            "<br><br>"
                            "<b>Manufacturing</b><br>"
                            "Shipments: %{customdata[0]:,.0f}<br>"
                            "Line-item value: $%{customdata[1]:,.0f}<br>"
                            "Freight cost: $%{customdata[2]:,.0f}"
                            "<br><br>"
                            "<b>Destination</b><br>"
                            "Shipments: %{customdata[3]:,.0f}<br>"
                            "Line-item value: $%{customdata[4]:,.0f}<br>"
                            "Freight cost: $%{customdata[5]:,.0f}"
                            "<extra></extra>"
                        ),

                        colorscale=scale_mfg,
                        showscale=True,

                        colorbar=dict(
                            title="Manufacturing",
                            x=1.02,
                            y=0.75,
                            len=0.4,
                        ),

                        marker_line_color="#ff4d4d",
                        marker_line_width=3,

                        name="Both",
                    )
                )

        # ========================================================
        # ORIGINAL 2D MAP
        # ========================================================

        fig.update_geos(
            showcountries=True,
            showframe=False,
            projection_type="natural earth",
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            height=520,
        )

        return fig


    # ============================================================
    # DISPLAY ONE MAP
    # ============================================================

    st.plotly_chart(
        draw_single_map(map_view),
        use_container_width=True,
    )

    st.markdown("#### Country rankings")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Top manufacturing countries**")
        st.dataframe(
            mfg.sort_values(metric_col, ascending=False)
            .drop(columns="iso3").head(15).round(0),
            use_container_width=True, hide_index=True,
        )
    with t2:
        st.markdown("**Top delivery countries**")
        st.dataframe(
            dest.sort_values(metric_col, ascending=False)
            .drop(columns="iso3").head(15).round(0),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Manufacturing country is taken from the **manufacturing_site.json** mapping.  \n"
        "Shipments with no identifiable site ('Not Applicable') are shown as Unknown "
        "and left off the map."
    )
