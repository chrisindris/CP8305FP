# Data Sources Documentation

**Project:** CP8305 – Knowledge Discovery Final Project  
**Team members:** _Add names here_  
**Date:** _Add date_

---

## Overview

This document describes all data sources used in the project, including their origin, relevance to the business problem, preprocessing steps, and any known limitations.

---

## 1. Primary Dataset

| Field | Details |
|-------|---------|
| **Name** | _Dataset name_ |
| **Source** | _URL or institution_ |
| **Access date** | _YYYY-MM-DD_ |
| **Format** | _CSV / Excel / JSON / API_ |
| **Size** | _Rows × Columns_ |
| **License** | _Public domain / CC-BY / etc._ |

### Description

_Provide a concise description of what this dataset contains and why it is relevant to the business problem._

### Key Variables

| Variable | Type | Description | Unit / Range |
|----------|------|-------------|-------------|
| `variable_1` | Numeric | _Description_ | _Unit_ |
| `variable_2` | Categorical | _Description_ | _Categories_ |
| `target` | _Type_ | _Description_ | _Values_ |

### Data Quality Notes

- **Missing values:** _Percentage and columns affected_
- **Duplicates:** _Number removed_
- **Outliers:** _Columns affected and treatment applied_
- **Class imbalance:** _If applicable_

---

## 2. Supplementary Dataset (if applicable)

| Field | Details |
|-------|---------|
| **Name** | _Dataset name_ |
| **Source** | _URL or institution_ |
| **Access date** | _YYYY-MM-DD_ |
| **Format** | _CSV / Excel / JSON / API_ |
| **Size** | _Rows × Columns_ |
| **License** | _Public domain / CC-BY / etc._ |

### Description

_Provide a concise description of what this dataset contains._

### How It Was Joined / Used

_Explain how this dataset was merged with the primary dataset or used independently._

---

## 3. Data Collection Process

_Describe how the data was collected. For example:_

- Data was downloaded directly from [Source Name] on [Date].
- API queries were made using the following parameters: …
- Data was collected in compliance with the source's terms of service.

---

## 4. Preprocessing Summary

| Step | Description | Script / Notebook |
|------|-------------|-------------------|
| Deduplication | Removed exact duplicate rows | `src/data_preprocessing.py` |
| Missing value imputation | Median for numeric; "Unknown" for categorical | `src/data_preprocessing.py` |
| Outlier removal | IQR method on selected columns | `src/data_preprocessing.py` |
| Encoding | One-hot encoding for categorical variables | `src/data_preprocessing.py` |
| Feature engineering | _Describe custom features added_ | `src/feature_engineering.py` |
| Scaling | StandardScaler applied inside modeling pipeline | `src/modeling.py` |

---

## 5. Citation

_Cite each data source following APA format. Example:_

> Author(s). (Year). *Dataset title* [Data set]. Publisher/Source. https://doi.org/xxxxx

---

## 6. Ethical Considerations

- The data used does not contain personally identifiable information (PII).
- _Add any other relevant ethical considerations._
