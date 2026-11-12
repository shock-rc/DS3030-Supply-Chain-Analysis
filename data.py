"""Data loading and reusable dashboard analysis helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DATA_FILE = DATA_DIR / "Supply_Chain_Shipment_Data.csv"
DATA_FILE = DATA_DIR / "Supply_Chain_Shipment_Data_Cleaned.csv"
MANUFACTURING_SITE_FILE = DATA_DIR / "manufacturing_site.json"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data.drop(columns="freight_status", inplace=True)
    return data


@st.cache_data
def calculate_pca(data):
    features = [
        "weight (kilograms)",
        "line item quantity",
        "line item value",
        "pack price",
        "unit price",
        "freight cost (usd)",
        "freight_per_kg",
    ]
    values = data[features].replace([np.inf, -np.inf], np.nan).dropna()
    scaled_values = StandardScaler().fit_transform(values)
    pca = PCA()
    transformed = pca.fit_transform(scaled_values)
    result = data.loc[values.index].copy()
    result["PC1"] = transformed[:, 0]
    result["PC2"] = transformed[:, 1]
    result["PC3"] = transformed[:, 2]
    return result, pca, features


@st.cache_data
def calculate_clusters(data, k):
    features = [
        "weight (kilograms)",
        "line item quantity",
        "line item value",
        "freight cost (usd)",
        "freight_per_kg",
        "freight_pct_of_value",
    ]
    values = data[features].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(scaled_values)
    result = data.loc[values.index].copy()
    result["cluster"] = labels
    silhouette = (
        silhouette_score(scaled_values, labels)
        if k > 1 and len(np.unique(labels)) > 1
        else np.nan
    )
    return result, model, scaler, features, silhouette


@st.cache_data
def association_rules(data, min_support=0.03, min_confidence=0.30):
    """Mine frequent categorical pairs and derive directional rules."""
    columns = ["product group", "shipment mode", "fulfill via", "vendor inco term"]
    work = data[columns].fillna("Unknown").astype(str)

    transactions = []
    for _, row in work.iterrows():
        transactions.append({f"{column}={row[column]}" for column in columns})

    item_counts = {}
    pair_counts = {}
    for items in transactions:
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1
        items_list = sorted(items)
        for index, first in enumerate(items_list):
            for second in items_list[index + 1:]:
                pair = (first, second)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    rows = []
    total_transactions = len(transactions)
    for (first, second), count in pair_counts.items():
        support = count / total_transactions
        if support < min_support:
            continue

        confidence_first = count / item_counts[first]
        confidence_second = count / item_counts[second]
        lift_first = confidence_first / (item_counts[second] / total_transactions)
        lift_second = confidence_second / (item_counts[first] / total_transactions)

        if confidence_first >= min_confidence:
            rows.append({
                "antecedent": first,
                "consequent": second,
                "support": support,
                "confidence": confidence_first,
                "lift": lift_first,
            })
        if confidence_second >= min_confidence:
            rows.append({
                "antecedent": second,
                "consequent": first,
                "support": support,
                "confidence": confidence_second,
                "lift": lift_second,
            })

    if not rows:
        return pd.DataFrame(
            columns=["antecedent", "consequent", "support", "confidence", "lift"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["lift", "confidence"], ascending=False)
        .drop_duplicates(["antecedent", "consequent"])
        .reset_index(drop=True)
    )
