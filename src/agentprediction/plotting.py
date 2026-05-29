from __future__ import annotations

import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.image import load_img
from scipy.spatial.distance import dice
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def lighten_color(color, amount: float = 0.5):
    """Lighten a matplotlib color by mixing it toward white."""

    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def find_best_k(data, max_k: int = 10) -> tuple[int, list[float]]:
    """Return the silhouette-optimal K and all tested scores."""

    scores = []
    k_values = range(2, max_k + 1)
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(data)
        scores.append(silhouette_score(data, labels))
    return list(k_values)[int(np.argmax(scores))], scores


def calculate_dice_coefficient(img1, img2, masker) -> float:
    data1 = masker.transform(load_img(img1)).astype(bool).ravel()
    data2 = masker.transform(load_img(img2)).astype(bool).ravel()
    return 1 - dice(data1, data2)


def create_cluster_mean_beta(masker, subjects_list, beta_dir, column_names) -> pd.DataFrame:
    """Create a subject x condition dataframe of cluster mean beta values."""

    rows = []
    for subject in subjects_list:
        row = {"subject": subject}
        for column in column_names:
            matches = list((beta_dir / subject).glob(f"*beta*{column}*.nii.gz"))
            if matches:
                row[column] = float(masker.transform(str(matches[0])).mean())
        rows.append(row)
    return pd.DataFrame(rows)
