from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_steady_state_outliers(columns_of_interest: list[str], all_columns) -> list[str]:
    """Add fMRIPrep outlier columns to the requested confound set."""

    outlier_columns = [column for column in all_columns if "outlier" in column]
    return [*columns_of_interest, *outlier_columns]


def create_confound_matrix(
    confound_file_path: str | Path,
    confounds_of_interest: list[str],
) -> pd.DataFrame:
    """Load a confounds TSV and replace missing values with zeros."""

    confounds = pd.read_csv(confound_file_path, sep="\t")
    columns = add_steady_state_outliers(confounds_of_interest, confounds.columns)
    return confounds[columns].fillna(0)
