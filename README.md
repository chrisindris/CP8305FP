# CP8305FP – Knowledge Discovery Final Project

## Team Members

| Name | Student ID |
|------|-----------|
| _Member 1_ | _ID_ |
| _Mojtaba Mohammadi_ | _501069345_ |
| _Member 3_ | _ID_ |

## Project Topic

_Brief description of the chosen industry/business and the business problem being addressed._

---

## Repository Structure

```
CP8305FP/
├── data/
│   ├── raw/                  # Original, immutable data files (not tracked by Git)
│   └── processed/            # Cleaned and transformed data (not tracked by Git)
├── docs/
│   └── data_sources.md       # Data sources documentation
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA – distributions, correlations, missing values
│   ├── 02_data_preprocessing.ipynb  # Cleaning, encoding, feature engineering
│   ├── 03_modeling.ipynb            # Model training, tuning, and selection
│   └── 04_analysis_findings.ipynb   # Results interpretation and recommendations
├── reports/
│   └── figures/              # Auto-generated charts and plots
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py # Data loading and cleaning utilities
│   ├── feature_engineering.py# Feature creation and scaling utilities
│   ├── modeling.py           # SKLearn model training, evaluation, and persistence
│   └── visualization.py      # Reusable Matplotlib/Seaborn plotting functions
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/chrisindris/CP8305FP.git
cd CP8305FP

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` folder in order:

1. **`01_data_exploration.ipynb`** – Explore the raw dataset.
2. **`02_data_preprocessing.ipynb`** – Clean and prepare the data.
3. **`03_modeling.ipynb`** – Train and evaluate SKLearn models.
4. **`04_analysis_findings.ipynb`** – Interpret results and build recommendations.

### Running the Preprocessing Script Directly

```bash
python -m src.data_preprocessing
```

---

## Workflow

```
data/raw/   →   02_data_preprocessing.ipynb   →   data/processed/
                         ↓
              03_modeling.ipynb  →  reports/ (saved model + figures)
                         ↓
              04_analysis_findings.ipynb  →  business recommendations
```

---

## Dependencies

Key libraries used in this project:

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Machine learning models, pipelines, and metrics |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisation |
| `plotly` | Interactive charts |
| `joblib` | Model serialisation |

See `requirements.txt` for the full list with pinned versions.

---

## Deliverables Checklist

- [ ] Comprehensive Final Project Report (PDF)
- [ ] Data Sources Documentation (`docs/data_sources.md`)
- [ ] Data Analysis and Modeling Artifacts (`src/`, `notebooks/`)
- [ ] Final Presentation (PPT + PDF)

---

## References

_List references here following APA format. Minimum five sources, including at least two journal papers._

1. …
2. …
3. …
4. …
5. …
