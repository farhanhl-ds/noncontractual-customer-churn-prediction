"""
Data cleaning and preprocessing utilities.

Handles:
- Loading UCI Online Retail II Excel sheets
- Merging both year sheets into a single DataFrame
- Removing cancelled transactions (InvoiceNo starting with 'C')
- Filtering missing CustomerIDs
- Handling negative quantities and prices
- Computing TotalPrice = Quantity x Price
"""

import pandas as pd


COLUMN_RENAME_MAP = {
    'Invoice'     : 'invoice',
    'StockCode'   : 'stock_code',
    'Description' : 'description',
    'Quantity'    : 'quantity',
    'InvoiceDate' : 'invoice_date',
    'Price'       : 'price',
    'Customer ID' : 'customer_id',
    'Country'     : 'country',
}


def load_raw(filepath: str) -> pd.DataFrame:
    """
    Load and concatenate both year sheets from the UCI Online Retail II Excel file.

    Parameters
    ----------
    filepath : str
        Path to online_retail_II.xlsx

    Returns
    -------
    pd.DataFrame
        Raw concatenated DataFrame with standardized column names.
    """
    df_0910 = pd.read_excel(filepath, sheet_name='Year 2009-2010')
    df_1011 = pd.read_excel(filepath, sheet_name='Year 2010-2011')

    df = pd.concat([df_0910, df_1011], ignore_index=True)
    df = df.rename(columns=COLUMN_RENAME_MAP)

    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['customer_id']  = df['customer_id'].astype('Int64')

    return df


def remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancellation transactions.

    Cancellations are identified by invoice numbers prefixed with 'C'.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with cancellation rows removed.
    """
    mask = ~df['invoice'].astype(str).str.startswith('C')
    n_removed = (~mask).sum()
    print(f"Removed {n_removed:,} cancellation rows ({n_removed / len(df) * 100:.1f}%)")
    return df[mask].copy()


def remove_invalid_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with non-positive quantity values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    mask = df['quantity'] > 0
    n_removed = (~mask).sum()
    print(f"Removed {n_removed:,} rows with non-positive quantity")
    return df[mask].copy()


def remove_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with non-positive price values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    mask = df['price'] > 0
    n_removed = (~mask).sum()
    print(f"Removed {n_removed:,} rows with non-positive price")
    return df[mask].copy()


def remove_missing_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows without a CustomerID (guest transactions).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    mask = df['customer_id'].notna()
    n_removed = (~mask).sum()
    print(f"Removed {n_removed:,} rows with missing customer_id ({n_removed / len(df) * 100:.1f}%)")
    return df[mask].copy()


def add_total_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a total_price column: quantity * price.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df['total_price'] = df['quantity'] * df['price']
    return df


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Full cleaning pipeline for UCI Online Retail II.

    Steps:
    1. Load and merge both year sheets
    2. Standardize column names and dtypes
    3. Remove cancellations
    4. Remove non-positive quantities
    5. Remove non-positive prices
    6. Remove missing customer IDs
    7. Add total_price column

    Parameters
    ----------
    filepath : str
        Path to online_retail_II.xlsx

    Returns
    -------
    pd.DataFrame
        Cleaned transaction DataFrame ready for RFM computation.
    """
    print("Loading data...")
    df = load_raw(filepath)
    print(f"Raw shape: {df.shape[0]:,} rows\n")

    print("Cleaning...")
    df = remove_cancellations(df)
    df = remove_invalid_quantities(df)
    df = remove_invalid_prices(df)
    df = remove_missing_customers(df)
    df = add_total_price(df)

    print(f"\nClean shape : {len(df):,} rows")
    print(f"Customers   : {df['customer_id'].nunique():,}")
    print(f"Date range  : {df['invoice_date'].min().date()} → {df['invoice_date'].max().date()}")

    return df