from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel

from agentprediction.config import AnalysisConfig


def collect_video_on_inputs(first_level_dir: Path, subjects: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Collect cue, feedback, and rest z-score images for the video-on model."""

    files: list[str] = []
    rows: list[dict[str, int]] = []
    patterns = [
        ("cue", "*z_score*cue*.nii.gz"),
        ("fb", "*z_score*fb*.nii.gz"),
        ("rest", "*z_score*rest*.nii.gz"),
    ]

    for label, pattern in patterns:
        for subject in subjects:
            matches = sorted((first_level_dir / subject).glob(f"sub-{subject}{pattern}"))
            if matches:
                files.append(str(matches[0]))
                rows.append(
                    {
                        "cue": int(label == "cue"),
                        "fb": int(label == "fb"),
                        "rest": int(label == "rest"),
                    }
                )

    return files, pd.DataFrame(rows, columns=["cue", "fb", "rest"])


def fit_video_on_second_level(
    config: AnalysisConfig,
    model_name: str = "model_VideoOn",
    subjects: list[str] | None = None,
) -> dict[str, object]:
    """Fit the group-level video-on contrasts and save all result images."""

    subjects = subjects or config.load_subjects()
    first_level_dir = config.first_level_dir(model_name)
    output_dir = config.second_level_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    second_level_input, design_matrix = collect_video_on_inputs(first_level_dir, subjects)
    second_level_model = SecondLevelModel().fit(second_level_input, design_matrix=design_matrix)
    plotting.plot_design_matrix(design_matrix, output_file=output_dir / "design_matrix.png")

    contrasts = {
        "video_rest_contrast": np.array([0.5, 0.5, -1]),
        "cue_fb_contrast": np.array([1, -1, 0]),
    }
    outputs = {}
    for contrast_name, contrast in contrasts.items():
        group_result = second_level_model.compute_contrast(contrast, output_type="all")
        outputs[contrast_name] = group_result
        for stats_name, image in group_result.items():
            nib.save(image, output_dir / f"{contrast_name}_{stats_name}.nii.gz")

    return {
        "model": second_level_model,
        "design_matrix": design_matrix,
        "inputs": second_level_input,
        "outputs": outputs,
    }


def threshold_z_map(
    z_map_path: str | Path,
    alpha: float = 0.05,
    height_control: str = "fdr",
    cluster_threshold: int = 30,
):
    return threshold_stats_img(
        nib.load(z_map_path),
        alpha=alpha,
        height_control=height_control,
        cluster_threshold=cluster_threshold,
    )
