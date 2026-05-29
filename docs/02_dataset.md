# Dataset

## UCI Online Retail II

- **Source**: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **Period**: December 2009 – December 2011
- **Raw records**: 1,067,371 transactions across both sheets
- **Unique customers (after cleaning)**: 5,789
- **Geography**: Primarily UK-based wholesaler

## Key Columns

| Column | Description |
|--------|-------------|
| `invoice` | Transaction ID. Prefix `C` = cancellation |
| `stock_code` | Product code (mixed type — numeric and alphanumeric e.g. `82494L`, `POST`) |
| `description` | Product description |
| `quantity` | Units purchased (negative = return) |
| `invoice_date` | Transaction timestamp |
| `price` | Unit price (GBP) |
| `customer_id` | Unique customer identifier (nullable — ~22.8% missing = guest transactions) |
| `country` | Customer country |

## Data Quality Summary

| Issue | Count | % of raw | Action |
|-------|-------|----------|--------|
| Missing `customer_id` | 243,007 | 22.8% | Dropped — untrackable guest transactions |
| Cancellation invoices (`C` prefix) | 19,494 | 1.8% | Dropped — reversed transactions |
| Non-positive quantity | 3,457 | 0.3% | Dropped — returns / adjustments |
| Non-positive price | 6,207 | 0.6% | Dropped — test or adjustment rows |
| Missing `description` | 4,382 | 0.4% | Retained — not used downstream |

**Clean dataset: 805,549 rows (75.5% of raw data retained)**

## Observation Window

We restrict to `2010-01-01 → 2011-12-09` (the actual last transaction date in the dataset).

> Note: The dataset's last transaction date is **December 9, 2011**, not December 31.
> The ~3-week gap at year-end is not missing data — it is the natural end of the source dataset.

| Period | Start | End | Rows | Customers |
|--------|-------|-----|------|-----------|
| Full observation | 2010-01-01 | 2011-12-09 | 774,795 | 5,789 |
| Calibration | 2010-01-01 | 2011-10-31 | 692,959 | 5,552 |
| Holdout | 2011-11-01 | 2011-12-09 | 81,836 | — |

## RFM Summary (Calibration Period)

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| Frequency (repeat purchases) | 4.76 | 2.0 | 0 | 345 |
| Recency (weeks) | 33.73 | 25.64 | 0 | 95 |
| T — customer age (weeks) | 61.67 | 69.29 | 0.14 | 94.86 |
| Monetary — avg order value (£) | 385.22 | 289.41 | 0.02 | 14,844.77 |

**Repeat buyer rate: 69.9%** (3,879 of 5,552 customers made at least 2 purchases in calibration)

## Acknowledged Limitations

- Dataset represents a **B2B wholesale** context — repeat purchase rates are higher and more
  regular than typical B2C marketplaces. Model generalizability to B2C settings (e.g. Tokopedia,
  Shopee) is limited and should be validated on domain-specific data.
- `stock_code` is a mixed-type column containing both numeric (`82494`) and alphanumeric
  (`82494L`, `POST`, `DOT`) values. Explicitly cast to `str` during preprocessing to avoid
  PyArrow type inference errors when writing to parquet.
- Frequency outlier at **345** — one customer made 346 purchases during calibration (~3.6/week).
  Likely a high-volume wholesale account. Results should be reported both with and without
  top 1% outlier buyers.
- Monetary distribution is heavily right-skewed (max £14,844 vs median £289).
  Gamma-Gamma CLV estimates may be disproportionately influenced by top spenders.
