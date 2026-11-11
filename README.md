
# Supply Chain Intelligence Dashboard

A Streamlit dashboard for the supplied Supply Chain Shipment dataset.

## Five required analytics functionalities

1. Data processing & data-quality analysis
2. Statistical analysis
3. PCA dimensionality reduction
4. K-Means clustering
5. Association-rule mining

An additional decision-oriented Insights page connects the analyses.

The Streamlit entry point is kept in `app.py`. Each dashboard page is implemented
in its own renderer under `pages/`, while shared loading and analysis helpers live
in `data.py`.

## Run locally

Create/activate a Python environment, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The CSV is included in this folder as `Supply_Chain_Shipment_Data.csv`.