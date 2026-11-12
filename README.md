# Supply Chain Financial & Delay Analysis Dashboard

An interactive Streamlit dashboard for exploring shipment value, freight cost,
delivery performance, supply-chain geography, and operational patterns in the
Supply Chain Shipment dataset.

The project combines a cleaned analytical dataset with interactive statistical
and machine-learning visualizations. It is designed for both exploratory
analysis and decision-oriented review of delivery delays, freight efficiency,
shipment groups, and country-level supply-chain activity.

## What the project provides

The dashboard includes the following analysis modules:

1. **Executive Overview**
   - Shipment count, line-item value, freight cost, late-delivery rate, and data completeness.
   - Freight cost by shipment mode.
   - Delivery-status distribution.
   - Shipment value versus freight cost.
   - Top destination countries by freight cost.

2. **Data Processing**
   - Raw-versus-cleaned row and column counts.
   - Missing-value and duplicate-row summaries.
   - Data-quality visualizations.
   - Explanation of the cleaning responses used for invalid numeric values,
     missing values, reference-based fields, and date fields.

3. **Statistical Profile & Cost Efficiency**
   - Distribution plots for numeric shipment and cost variables.
   - Categorical comparisons.
   - Delivery-delay plots.
   - Cost-per-unit and freight-per-value analysis.
   - Identification of high-cost shipments for review.

4. **Delivery Delay Analysis**
   - Overall delay statistics and delivery-status distribution.
   - Delay distributions by shipment mode.
   - Destination-level delay comparisons.
   - Delay trends over time.
   - A review table for the most delayed shipments.

5. **Supply Map**
   - Manufacturing and destination-country views.
   - Combined origin-and-destination visualizations.
   - Country rankings for manufacturing and delivery activity.
   - Manufacturing-site lookup using `data/manufacturing_site.json`.

6. **Principal Component Analysis (PCA)**
   - Selectable numeric features.
   - Log transformations for skewed positive variables.
   - Signed log transformation for delivery delay.
   - Standardization, explained-variance analysis, and component loadings.
   - Three-dimensional PCA visualization.

7. **K-Means and K-Prototypes Clustering**
   - Numeric feature preparation and standardization.
   - K-Means model evaluation using elbow and silhouette analysis.
   - PCA-space cluster visualization.
   - Cluster-size, profile, boxplot, categorical, country, year, and late-delivery
     comparisons.
   - K-Prototypes support for mixed numeric and categorical analysis.

8. **Association Rules**
   - Correlation matrix for numeric variables.
   - Frequent categorical-pair mining across product group, shipment mode,
     fulfillment method, and vendor incoterm.
   - Adjustable minimum support and confidence.
   - Rule ranking by support, confidence, and lift.

9. **Insights & Hypothesis Testing**
   - Chi-square test of freight-cost category versus delivery-delay category.
   - Welch's independent t-test comparing historical delivery-delay periods.
   - Statistical interpretation, contingency tables, and priority-review summaries.

All dashboard pages use the sidebar filters for destination country, shipment
mode, product group, and manager. Filters are applied consistently to the
selected analysis module.

## Technology stack

- Python
- Streamlit
- pandas
- NumPy
- Plotly
- SciPy
- scikit-learn
- kmodes
- Jupyter notebooks for the original analysis workflow

The exact package requirements are listed in
[requirements.txt](requirements.txt).

## Repository structure

```text
.
├── app.py                         # Streamlit application entry point
├── data.py                        # Data paths, loading, PCA, clustering, and rules
├── requirements.txt               # Python dependencies
├── data/
│   ├── Supply_Chain_Shipment_Data.csv
│   ├── Supply_Chain_Shipment_Data_Cleaned.csv
│   └── manufacturing_site.json
├── pages/
│   ├── overview.py                # Executive overview
│   ├── data_processing.py         # Data-quality analysis
│   ├── statistics.py              # Statistical and cost analysis
│   ├── delay_analysis.py          # Delivery-delay analysis
│   ├── supply_map.py              # Geographic analysis
│   ├── pca.py                     # PCA and PCA-based clustering
│   ├── clustering.py              # Cluster analysis
│   ├── association_rules.py       # Association-rule analysis
│   ├── insights.py                # Hypothesis testing and conclusions
│   └── common.py                 # Shared third-party imports
└── initial_analysis/
    ├── data_cleaning_pipeline.ipynb
    ├── main.ipynb
    └── pca_clustering.ipynb
```

## Requirements

- Python 3.10 or newer
- `pip`
- A supported operating system with a terminal

The dashboard reads its data from the repository-relative `data/` directory,
so it can be launched from any working directory after the repository is
checked out.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/shock-rc/DS3030-Supply-Chain-Analysis.git
cd DS3030-Supply-Chain-Analysis
```

Create and activate a virtual environment:

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the dashboard

From the repository root, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL, normally:

```text
http://localhost:8501
```

Open that URL in a browser. Select an analysis module using the page selector
and optionally narrow the results with the sidebar filters.

## Working with the data

The repository includes two CSV files:

- `data/Supply_Chain_Shipment_Data.csv` — source/raw shipment data used for
  data-quality comparisons.
- `data/Supply_Chain_Shipment_Data_Cleaned.csv` — cleaned dataset used by the
  dashboard.

The dashboard loads the cleaned CSV through `data.py`. The data loader removes
the intermediate `freight_status` column before rendering the analysis pages.
The data-processing page reads the raw CSV to compare completeness, shape, and
duplicates with the cleaned data.

The file `data/manufacturing_site.json` maps manufacturing sites to countries
for the supply-chain map. Keep this file alongside the CSVs when moving or
deploying the project.

## Re-running the notebooks

The `initial_analysis/` directory contains the project’s exploratory notebooks:

- `data_cleaning_pipeline.ipynb` documents the cleaning workflow.
- `main.ipynb` contains the primary exploratory analysis.
- `pca_clustering.ipynb` contains PCA and clustering exploration.

To open them locally:

```bash
python -m pip install jupyter
jupyter notebook
```

Then open the required notebook from `initial_analysis/`.

## Validation and development checks

The repository does not currently include an automated test suite. The
following smoke checks validate the main Python code and reusable analysis
helpers:

```bash
python -m compileall -q app.py data.py pages
python -c "from data import load_data; print(load_data().shape)"
```

For a complete application check, launch the Streamlit dashboard and open each
analysis module using both the unfiltered dataset and representative sidebar
filters.

## Notes

- Analysis results depend on the selected filters and the available data.
- PCA, clustering, and association-rule views may exclude rows with missing or
  invalid values required by the selected calculation.
- Statistical tests show associations in this dataset; they should not be
  interpreted as proof of causation.
- Streamlit caching is used for reusable data and model calculations to keep
  interactive navigation responsive.
