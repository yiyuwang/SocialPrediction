from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import concat_imgs, mean_img, new_img_like
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from agentprediction.config import AnalysisConfig
from agentprediction.plotting import lighten_color


ANOVA_CONDITIONS = [
    "obs_Pattern",
    "obs_Social",
    "fb_Pattern_Congruent",
    "fb_Social_Congruent",
    "fb_Pattern_PE",
    "fb_Social_PE",
]

OMNIBUS_CONTRAST = np.array(
    [
        [0, -1, 0, 0, 0, 1],
        [0, 0, -1, 0, 0, 1],
        [0, 0, 0, -1, 0, 1],
        [0, 0, 0, 0, -1, 1],
        [-1, 0, 0, 0, 0, 1],
    ]
)


def collect_anova_inputs(
    beta_dir: str | Path,
    subjects: list[str],
    conditions: list[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Collect one z-map per subject and condition for the omnibus ANOVA."""

    beta_dir = Path(beta_dir)
    conditions = conditions or ANOVA_CONDITIONS
    files = []
    rows = []

    for condition in conditions:
        for subject in subjects:
            matches = sorted((beta_dir / subject).glob(f"sub-{subject}*z_score*{condition}*.nii.gz"))
            if not matches:
                print(f"no {condition} for subject {subject}")
                continue

            files.append(str(matches[0]))
            rows.append({name: int(name == condition) for name in conditions})

    return files, pd.DataFrame(rows, columns=conditions)


def fit_omnibus_anova(
    config: AnalysisConfig,
    model: str,
    subjects: list[str],
    conditions: list[str] | None = None,
    contrast: np.ndarray = OMNIBUS_CONTRAST,
) -> dict[str, object]:
    """Fit the shared second-level omnibus ANOVA and save result maps."""

    beta_dir = config.first_level_dir(model)
    output_dir = config.path(config.results_dir) / model / "omnibus_anova_F"
    output_dir.mkdir(parents=True, exist_ok=True)

    second_level_input, design_matrix = collect_anova_inputs(beta_dir, subjects, conditions)
    second_level_model = SecondLevelModel().fit(second_level_input, design_matrix=design_matrix)
    plotting.plot_design_matrix(design_matrix, output_file=output_dir / "design_matrix.png")

    print(f"Rank of the omnibus contrast matrix: {np.linalg.matrix_rank(contrast)}")
    group_res = second_level_model.compute_contrast(
        contrast,
        output_type="all",
        second_level_stat_type="F",
    )
    for stats_name, image in group_res.items():
        out_file = output_dir / f"omnibus_anova_{stats_name}.nii.gz"
        print(f"saving {out_file}")
        nib.save(image, out_file)

    return {
        "beta_dir": beta_dir,
        "second_level_res_dir": output_dir,
        "second_level_input": second_level_input,
        "design_matrix": design_matrix,
        "second_level_model": second_level_model,
        "group_res": group_res,
    }


def find_best_k(data, max_k: int = 10):
    inertia = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    inertia_diff = np.diff(inertia)
    inertia_diff2 = np.diff(inertia_diff)
    elbow_k = int(np.argmax(inertia_diff2) + 3)
    best_silhouette_k = int(np.argmax(silhouette_scores) + 2)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia, "bo-", label="Inertia")
    plt.axvline(x=elbow_k, color="g", linestyle="--", label=f"Elbow at k={elbow_k}")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, "ro-", label="Silhouette Score")
    plt.axvline(
        x=best_silhouette_k,
        color="g",
        linestyle="--",
        label=f"Best k={best_silhouette_k}",
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Best k by Elbow method: {elbow_k}")
    print(f"Best k by Silhouette method: {best_silhouette_k}")
    return {"best_k_elbow": elbow_k, "best_k_silhouette": best_silhouette_k}


def plot_knee_curve(data, max_k: int = 12, random_state: int = 42, title: str = "Full Sample"):
    from kneed import KneeLocator
    from sklearn.cluster import KMeans

    k_values = range(1, max_k + 1)
    inertias = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    knee = KneeLocator(
        x=list(k_values),
        y=inertias,
        curve="convex",
        direction="decreasing",
    )
    optimal_k = knee.elbow
    print("Optimal number of clusters:", optimal_k)

    plt.rcParams.update({"font.family": "Arial", "font.size": 12})
    plt.figure(figsize=(5, 5))
    plt.plot(k_values, inertias, "bo-")
    if optimal_k is not None:
        plt.vlines(optimal_k, ymin=min(inertias), ymax=max(inertias), linestyles="dashed", colors="red")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.title(title)
    plt.legend()
    plt.show()
    return optimal_k, inertias


def create_cluster_mean_beta(
    masker,
    beta_dir: str | Path,
    column_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cluster mean and SEM z-scores for each condition."""

    beta_dir = Path(beta_dir)
    column_names = column_names or ANOVA_CONDITIONS
    cluster_mean_beta = []
    cluster_std_beta = []

    for column in column_names:
        file_list = sorted(beta_dir.glob(f"*/sub-*z_score*{column}*.nii.gz"))
        img = concat_imgs([str(path) for path in file_list])
        beta_avg = mean_img(img)
        beta_mean = masker.fit_transform(beta_avg)
        cluster_mean_beta.append(beta_mean.T)

        beta_std = np.std(img.get_fdata(), axis=-1)
        standard_error = beta_std / np.sqrt(img.shape[-1])
        beta_sem = masker.fit_transform(new_img_like(beta_avg, standard_error))
        cluster_std_beta.append(beta_sem.T)

    cluster_mean_beta = np.array(cluster_mean_beta).squeeze()
    cluster_std_beta = np.array(cluster_std_beta).squeeze()
    df = pd.DataFrame(cluster_mean_beta.T, columns=column_names)
    df_std = pd.DataFrame(cluster_std_beta.T, columns=column_names)

    plot_names = [
        "Cue-AI",
        "Cue-AD",
        "Feedback Congruent AI",
        "Feedback Congruent AD",
        "Feedback Pred Error AI",
        "Feedback Pred Error AD",
    ]
    if len(column_names) == len(plot_names):
        df.columns = plot_names
        df_std.columns = plot_names
    return df, df_std


def create_bar_plots_for_clusters(
    df,
    df_sem=None,
    title: str = "",
    hatches=None,
    color_list=None,
    ylabel: str = "Mean Activation (Z-score)",
):
    num_clusters = len(df)
    y_min = df.min().min() - 0.5
    y_max = df.max().max() + 0.5

    if num_clusters < 10:
        n_rows = 1
        n_cols = num_clusters
        fig_size = (21, 7)
    else:
        n_rows = 4
        n_cols = math.ceil(num_clusters / n_rows)
        fig_size = (21, 14)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    axes = np.array(axes).flatten()
    color_list = color_list or ["gray"] * num_clusters
    hatches = hatches or ["xx", "++", "//", "//||", ".", ".||"]
    num_features = len(df.columns)

    for i in range(num_clusters):
        x = np.arange(num_features)
        bars = axes[i].bar(
            x,
            df.iloc[i],
            color=color_list[i],
            width=0.6,
            yerr=None if df_sem is None else df_sem.iloc[i],
            capsize=5,
            error_kw={"ecolor": "black", "linewidth": 1},
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        axes[i].set_title(f"Group {i + 1}", fontsize=16, fontweight="bold")
        axes[i].tick_params(axis="x", which="both", bottom=i >= num_clusters - n_cols, top=False, labelbottom=False)
        axes[i].tick_params(axis="y", which="major", labelsize=14, width=2, length=6)
        axes[i].set_ylabel(ylabel if i % n_cols == 0 else "", fontsize=16, fontweight="bold")
        axes[i].set_ylim([y_min, y_max])
        axes[i].grid(False)
        axes[i].spines["left"].set_linewidth(2)
        axes[i].spines["bottom"].set_linewidth(2)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    for j in range(num_clusters, n_rows * n_cols):
        fig.delaxes(axes[j])

    labels = [label.replace("_", " ") for label in df.columns]
    legend_elements = [
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", hatch=hatch, label=label, linewidth=0.5)
        for hatch, label in zip(hatches, labels)
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=num_features, fontsize=24)
    fig.suptitle(title, fontsize=16)
    plt.show()
    plt.clf()
    plt.close()
    return fig


def create_grouped_barplot(data, groups):
    n_cols = len(groups)
    fig, axes = plt.subplots(1, n_cols, figsize=(20, 7))
    axes = np.array(axes).ravel()
    width = 0.25
    group_colors = ["#FFC800", "#7A3EB1", "#F76800"]
    videos = ["obs", "Congruent", "PE"]

    all_values = []
    all_sems = []
    for group in groups:
        for video in videos:
            for condition in data["condition"].unique():
                subset = data[(data["condition"] == condition) & (data["video"] == video)]
                all_values.append(subset[group].mean())
                all_sems.append(subset[group].sem())
    max_value = max(all_values) + max(all_sems) + 0.1
    min_value = min(all_values) - max(all_sems) - 0.1

    for idx, group in enumerate(groups):
        ax = axes[idx]
        light_color = lighten_color(group_colors[idx], 0.6)
        unique_conditions = data["condition"].unique()
        x = np.arange(len(unique_conditions))

        for i, video in enumerate(videos):
            means = []
            sems = []
            for condition in unique_conditions:
                subset = data[(data["condition"] == condition) & (data["video"] == video)]
                means.append(subset[group].mean())
                sems.append(subset[group].sem())

            hatch = "" if video == "obs" else "x" if video == "PE" else "."
            ax.bar(
                x + (i - 1) * width,
                means,
                width,
                label=video,
                color=light_color,
                hatch=hatch,
                yerr=sems,
                capsize=5,
                edgecolor="black",
                linewidth=1,
            )

        ax.set_ylabel("Mean Parameter Estimate (Z-scored)", fontsize=16, fontweight="bold")
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim([min_value, max_value])
        ax.set_title(f"{group}", fontsize=26, pad=10, fontweight="bold", font="Arial")
        ax.set_xticks(x)
        ax.set_xticklabels(unique_conditions, fontsize=40, font="Arial")
        ax.tick_params(axis="both", which="major", labelsize=20)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    return fig


# Backward-compatible aliases for older notebooks.
CreateClusterMeanBeta = create_cluster_mean_beta
CreateBarPlotsForClusters = create_bar_plots_for_clusters
