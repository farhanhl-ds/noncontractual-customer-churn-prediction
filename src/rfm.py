"""
RFM computation and calibration/holdout split utilities.

Computes per-customer RFM metrics in the format expected by the BG/NBD model:

    - frequency      : number of repeat transactions (distinct invoices - 1)
    - recency        : time between first and last purchase (in weeks)
    - T              : time between first purchase and observation end date (in weeks)
    - monetary_value : average revenue per transaction (excluding first)

Note on BG/NBD frequency convention:
    BG/NBD defines frequency as the count of *repeat* purchases, not total purchases.
    A customer who bought once has frequency=0 and is valid input to the model.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calibration_holdout_split(
    df: pd.DataFrame,
    holdout_start: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a cleaned transaction DataFrame into calibration and holdout periods.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame with an 'invoice_date' column.
    holdout_start : pd.Timestamp
        First date of the holdout period. All transactions on or after
        this date go into the holdout set.

    Returns
    -------
    df_calib : pd.DataFrame
        Transactions strictly before holdout_start.
    df_holdout : pd.DataFrame
        Transactions from holdout_start onward.
    """
    df_calib   = df[df['invoice_date'] < holdout_start].copy()
    df_holdout = df[df['invoice_date'] >= holdout_start].copy()

    print(f"Calibration : {len(df_calib):,} rows  "
          f"({df_calib['invoice_date'].min().date()} → {df_calib['invoice_date'].max().date()})")
    print(f"Holdout     : {len(df_holdout):,} rows  "
          f"({df_holdout['invoice_date'].min().date()} → {df_holdout['invoice_date'].max().date()})")

    return df_calib, df_holdout


def compute_rfm(
    df: pd.DataFrame,
    observation_end: pd.Timestamp,
    time_unit: str = 'W',
) -> pd.DataFrame:
    """
    Compute per-customer RFM metrics in BG/NBD format.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame. Must contain:
        'customer_id', 'invoice', 'invoice_date', 'total_price'
    observation_end : pd.Timestamp
        The end of the observation window. Recency and T are measured
        relative to this date.
    time_unit : str
        Pandas offset alias for time unit. Default 'W' (weeks).
        Use 'D' for days if needed.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        customer_id, frequency, recency, T, monetary_value
    """
    # Aggregate to invoice level first (one row per invoice per customer)
    invoices = (
        df
        .groupby(['customer_id', 'invoice', 'invoice_date'])['total_price']
        .sum()
        .reset_index()
        .rename(columns={'total_price': 'invoice_value'})
    )

    # Per-customer aggregation
    rfm = (
        invoices
        .groupby('customer_id')
        .agg(
            first_purchase=('invoice_date', 'min'),
            last_purchase=('invoice_date', 'max'),
            total_invoices=('invoice', 'nunique'),
            avg_invoice_value=('invoice_value', 'mean'),
        )
        .reset_index()
    )

    # Convert time deltas to weeks (or chosen unit)
    unit_map = {'W': 7, 'D': 1}
    days_per_unit = unit_map.get(time_unit, 7)

    rfm['recency'] = (
        (rfm['last_purchase'] - rfm['first_purchase']).dt.days / days_per_unit
    ).round(2)

    rfm['T'] = (
        (observation_end - rfm['first_purchase']).dt.days / days_per_unit
    ).round(2)

    # BG/NBD frequency = repeat purchases = total_invoices - 1
    rfm['frequency'] = rfm['total_invoices'] - 1

    # Monetary value: average transaction value
    # For customers with frequency=0, this is their single purchase value
    rfm['monetary_value'] = rfm['avg_invoice_value'].round(2)

    # Drop customers where T <= 0 (edge case: first purchase on observation_end)
    rfm = rfm[rfm['T'] > 0].copy()

    # Select and order final columns
    rfm = rfm[[
        'customer_id', 'frequency', 'recency', 'T', 'monetary_value',
        'first_purchase', 'last_purchase', 'total_invoices'
    ]]

    print(f"RFM computed for {len(rfm):,} customers")
    print(f"  frequency  — mean: {rfm['frequency'].mean():.2f}, max: {rfm['frequency'].max()}")
    print(f"  recency    — mean: {rfm['recency'].mean():.2f} {time_unit}")
    print(f"  T          — mean: {rfm['T'].mean():.2f} {time_unit}")
    print(f"  monetary   — mean: £{rfm['monetary_value'].mean():.2f}")

    return rfm


def get_repeat_customers(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Filter RFM table to customers with at least one repeat purchase.

    Used for Gamma-Gamma model fitting, which requires frequency > 0.

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of compute_rfm().

    Returns
    -------
    pd.DataFrame
        Subset where frequency >= 1.
    """
    repeat = rfm[rfm['frequency'] >= 1].copy()
    print(f"Repeat customers (frequency >= 1): {len(repeat):,} "
          f"({len(repeat) / len(rfm) * 100:.1f}% of total)")
    return repeat
