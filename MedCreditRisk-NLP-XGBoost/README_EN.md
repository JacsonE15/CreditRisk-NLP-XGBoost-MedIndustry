# Credit Risk Prediction Model for China's Pharmaceutical Industry

A machine learning pipeline that predicts whether A-share listed pharmaceutical companies will receive Special Treatment (ST) designation within the next 3 years, using financial statement data and NLP sentiment analysis of annual report narratives.

---

## Background

The ST (Special Treatment) system is a risk-warning mechanism in China's A-share market, applied to companies with financial irregularities or compliance violations. This project combines structured financial data with unstructured text features extracted from the Management Discussion & Analysis (MDA) section of annual reports to build an early-warning model for ST risk.

---

## Pipeline Overview

```
Financial Data Scraping → Annual Report PDF Download → MDA Text Extraction → FinBERT Sentiment Analysis → Feature Engineering → Model Training → Evaluation
```

---

## Repository Structure

```
credit_risk_project/
│
├── 1_Download_3_finance_table.ipynb        # Step 1: Scrape financial statements + industry classification
├── 2_Download_annual_report_and_NLP.ipynb  # Step 2: Download annual report PDFs + MDA extraction + NLP
├── 3_Training_and_result.ipynb             # Step 3: Feature engineering + model training + visualization
│
├── master2.csv                             # Final wide table (financial + NLP features merged)
└── BDT_CO.csv                              # ST event data for pharmaceutical industry
```

---

## Step-by-Step Description

### Notebook 1: Financial Data Collection

**Data Source:** AKShare (EastMoney API)

**Tables Collected:**
- Income Statement (`stock_lrb_em`)
- Balance Sheet (`stock_zcfz_em`)
- Cash Flow Statement (`stock_xjll_em`)
- Shenwan Level-1 Industry Classification

**Coverage:** All A-share listed companies, fiscal years 2020–2025

**Filter:** Shenwan Level-1 Industry = Pharmaceuticals & Biotech → **451 companies**

**Output:** `master_table.csv`

---

### Notebook 2: Annual Report Download & NLP Analysis

**PDF Download:**
- Retrieve annual report PDF links via AKShare and download in batches (~100 companies per batch)
- Coverage: fiscal years 2020–2025

**MDA Text Extraction:**
- Parse PDFs using `pdfplumber`
- Locate the start and end of the "Management Discussion & Analysis" section by keyword matching
- Automatically skip table-of-contents pages and scanned (non-text) PDFs

**NLP Sentiment Analysis:**
- Model: `yiyanghkust/finbert-tone` (FinBERT fine-tuned for financial sentiment)
- Features extracted:
  - `negative_score`: mean negative sentiment probability across MDA sentences
  - `negative_score_max`: maximum negative sentiment probability
  - `risk_word_ratio`: proportion of risk-related keywords
  - `uncertainty_ratio`: proportion of uncertainty-related keywords
  - `mda_length`: total length of the MDA section

**Output:** `master2.csv` (financial + NLP features merged into a single wide table)

---

### Notebook 3: Model Training & Evaluation

#### Label Definition (Y)

```
Y = 1 : The company receives its first ST designation within 3 years after the reporting period
Y = 0 : No ST event occurs within the 3-year window
```

> Observations from the year of ST designation onward are excluded to prevent data leakage.

**Sample Statistics:**
- Total samples: 2,168
- Positive samples (Y=1): 16 (0.74%)
- ST reasons: Audit denial (8), Two consecutive loss years (3), Disclosure violation (3), Major litigation (2)

#### Feature Engineering

| Category | Features |
|----------|----------|
| Scale | `log_Revenue`, `log_TotalAssets` |
| Profitability | `NetProfitMargin`, `OperatingMargin`, `ROE`, `log_NetProfit` |
| Cash Flow | `CFO_to_Assets`, `CashRatio`, `CashProfitQuality` |
| Leverage | `DebtRatio` |
| Growth | YoY growth rates for net profit, revenue, operating profit, total assets |
| Trend | t-1 lagged features for margins, ROE, and cash flow ratios |
| NLP | `negative_score`, `negative_score_max`, `log_risk_word_ratio`, `log_uncertainty_ratio`, `log_mda_length` |

**Preprocessing:**
- Replace `inf` / `-inf` with `NaN`; fill missing values with global median
- Winsorize at 5%–95% to handle extreme outliers

#### Validation Strategy

Expanding-window time-series cross-validation (no future data leakage):

```
Fold 1: Train 2020           → Validate 2021
Fold 2: Train 2020–2021      → Validate 2022
Fold 3: Train 2020–2022      → Validate 2023
Fold 4: Train 2020–2023      → Validate 2024
```

#### Models

- **Logistic Regression** — `class_weight='balanced'`
- **XGBoost** — `scale_pos_weight` to handle class imbalance
- **LightGBM** — early stopping + `scale_pos_weight`
- **Ensemble** — probability average of the three models above

---

## Results

| Model | Mean ROC-AUC | Mean PR-AUC |
|-------|-------------|------------|
| Logistic Regression | 0.563 | 0.026 |
| XGBoost | **0.880** | **0.121** |
| LightGBM | 0.751 | 0.056 |
| Ensemble | 0.728 | 0.037 |

> Random baseline PR-AUC ≈ 0.007 (equal to the positive class ratio)
> XGBoost PR-AUC is approximately **17× above the random baseline**

---

## Limitations

1. **Very few positive samples** — only 16 ST events across 2,168 observations; each validation fold contains only 2–4 positives, making results sensitive to individual predictions
2. **Unstable early folds** — Fold 1 has only 3 training positives, so results from Fold 3 and Fold 4 are more reliable
3. **Diverse ST triggers** — events such as audit denial and disclosure violations are weakly correlated with financial ratios, limiting predictability from structured data alone
4. **NLP model limitations** — the FinBERT model used is a Chinese transfer of an English financial model, which may not fully capture the linguistic patterns of A-share annual reports
5. **Scalability** — the pipeline is designed to scale; extending to all A-share industries would substantially increase positive samples and is expected to raise PR-AUC to the 0.3–0.5 range

---

## Dependencies

```bash
pip install akshare pdfplumber transformers torch sentencepiece
pip install pandas numpy scipy scikit-learn xgboost lightgbm matplotlib
```

**Runtime:** Google Colab + Google Drive

---

## Data Description

| File | Description |
|------|-------------|
| `master2.csv` | Wide table with financial and NLP features for 451 pharmaceutical companies, 2020–2025 (2,333 rows) |
| `BDT_CO.csv` | ST event records for the pharmaceutical industry, including first ST date, ST reason, and survival time (519 rows) |
