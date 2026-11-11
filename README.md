
# Supply Chain Intelligence Dashboard

A Streamlit dashboard for the supplied Supply Chain Shipment dataset.

## Five required analytics functionalities

1. Data processing & data-quality analysis
2. Statistical analysis
3. PCA dimensionality reduction
4. K-Means clustering
5. Association-rule mining

An additional decision-oriented Insights page connects the analyses.

## Run locally

Create/activate a Python environment, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The CSV is included in this folder as `Supply_Chain_Shipment_Data.csv`.

## Suggested viva storyline

**Clean → Measure → Reduce → Group → Discover → Explain**

- Clean inconsistent numeric/date fields and preserve meaningful missingness.
- Measure cost, freight efficiency and delivery performance.
- Reduce correlated shipment variables with PCA.
- Group shipment behaviours with K-Means and validate K with silhouette score.
- Discover recurring combinations with association rules.
- Use the Insights page to identify shipments worth operational investigation.

## Important interpretation note

The dashboard's high-cost / high-freight-per-unit rows are **investigation flags**, not proof that a shipment was operationally wrong. Likewise, association rules describe co-occurrence, not causation.
