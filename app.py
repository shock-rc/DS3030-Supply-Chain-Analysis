import streamlit as st

from data import load_data
from pages.association_rules import render as render_association_rules
from pages.clustering import render as render_clustering
from pages.data_processing import render as render_data_processing
from pages.delay_analysis import render as render_delay_analysis
from pages.insights import render as render_insights
from pages.overview import render as render_overview
from pages.pca import render as render_pca
from pages.statistics import render as render_statistics
from pages.supply_map import render as render_supply_map


st.set_page_config(
    page_title="Supply Chain Financial & Delay Analysis Dashboard",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("DS3030 : Supply Chain Financial & Delay Analysis Dashboard")
st.caption("The cleaned data used for the dashboard can be referenced at : https://github.com/shock-rc/DS3030-Supply-Chain-Analysis/blob/main/Supply_Chain_Shipment_Data_Cleaned.csv")


def build_sidebar_filters(data):
    """Render the sidebar controls and return the selected filters."""
    st.sidebar.header("Dashboard Controls")
    return {
        "country": st.sidebar.multiselect(
            "Destination country",
            sorted(data["country"].dropna().unique()),
            default=[],
        ),
        "shipment mode": st.sidebar.multiselect(
            "Shipment mode",
            sorted(data["shipment mode"].dropna().unique()),
            default=[],
        ),
        "product group": st.sidebar.multiselect(
            "Product group",
            sorted(data["product group"].dropna().unique()),
            default=[],
        ),
        "managed by": st.sidebar.multiselect(
            "Managed by",
            sorted(data["managed by"].dropna().unique()),
            default=[],
        ),
    }


def apply_filters(data, filters):
    """Return the rows matching all selected sidebar filters."""
    filtered = data.copy()
    for column, selected_values in filters.items():
        if selected_values:
            filtered = filtered[filtered[column].isin(selected_values)]
    return filtered


def select_page():
    """Render the page selector and return the selected dashboard page."""
    return st.pills(
        "Analysis module",
        [
            "Overview",
            "Data Processing",
            "Statistics",
            "Delay Analysis",
            "Supply Map",
            "PCA",
            "Clustering",
            "Association Rules",
            "Insights & Hypothesis Testing",
        ],
        default="Data Processing",
        label_visibility="collapsed",
    )


def main():
    data = load_data()
    filters = build_sidebar_filters(data)
    filtered = apply_filters(data, filters)

    renderers = {
        "Overview": render_overview,
        "Data Processing": render_data_processing,
        "Statistics": render_statistics,
        "Delay Analysis": render_delay_analysis,
        "Supply Map": render_supply_map,
        "PCA": render_pca,
        "Clustering": render_clustering,
        "Association Rules": render_association_rules,
        "Insights & Hypothesis Testing": render_insights,
    }
    selected_page = select_page()
    renderers.get(selected_page, render_overview)(filtered, data)


if __name__ == "__main__":
    main()
