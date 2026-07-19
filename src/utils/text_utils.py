"""Shared text normalization used by training and evaluation."""

import pandas as pd


def has_real_input(values: pd.Series) -> pd.Series:
    normalized = (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
    )
    return normalized.ne("<noinput>") & normalized.ne("")
