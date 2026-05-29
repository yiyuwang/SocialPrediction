from __future__ import annotations

import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs

from agentprediction.confounds import create_confound_matrix
from agentprediction.config import AnalysisConfig
from agentprediction.events import create_events_dataframe


def find_task_file(logfiles_dir: Path, subject: str) -> Path:
    matches = sorted(logfiles_dir.glob(f"*{subject}*edited.txt"))
    if not matches:
        raise FileNotFoundError(f"No edited logfile found for subject {subject} in {logfiles_dir}")
    return matches[0]


def fit_run_design_matrices(
    config: AnalysisConfig,
    model_name: str,
    subjects: list[str],
    event_parser,
    event_parser_kwargs: dict | None = None,
    runs: tuple[int, ...] = (1, 2),
    create_design_matrix: bool = True,
) -> list[pd.DataFrame]:
    """Fit and save per-run first-level design matrices."""

    output_dir = config.first_level_dir(model_name)
    event_parser_kwargs = event_parser_kwargs or {}

    if not create_design_matrix:
        return load_design_matrices(output_dir, subjects, runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    design_matrices = []
    for subject in subjects:
        print(f"running subject {subject}")
        subject_output_dir = output_dir / subject
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        task_file = find_task_file(config.path(config.logfiles_dir), subject)

        for run in runs:
            events = create_events_dataframe(task_file, run, event_parser, **event_parser_kwargs)
            confound_file = (
                config.path(config.confounds_dir)
                / f"sub-{subject}_task-socialpred_run-{run}_desc-confounds_timeseries.tsv"
            )
            cov = create_confound_matrix(confound_file, config.confounds_of_interest)

            fmri_glm = FirstLevelModel(
                t_r=config.tr,
                noise_model="ar3",
                standardize=True,
                hrf_model="spm",
                drift_model="cosine",
                high_pass=0.012,
                mask_img=config.gm_mask_img,
                smoothing_fwhm=6,
            )
            func_path = config.path(config.data_dir) / f"sub-{subject}_socialpred_run{run}.nii.gz"
            fmri_glm = fmri_glm.fit(nib.load(func_path), events, confounds=cov)

            design_matrix = fmri_glm.design_matrices_[0]
            design_matrices.append(design_matrix)
            plotting.plot_design_matrix(
                design_matrix,
                output_file=subject_output_dir / f"design_matrix_run{run}.png",
            )
            design_matrix.to_csv(subject_output_dir / f"sub-{subject}_run-{run}_design_matrix.csv")

    with open(output_dir / "design_matrices.pkl", "wb") as handle:
        pickle.dump(design_matrices, handle)

    return design_matrices


def load_design_matrices(
    first_level_dir: Path,
    subjects: list[str],
    runs: tuple[int, ...] = (1, 2),
) -> list[pd.DataFrame]:
    matrices = []
    for subject in subjects:
        subject_output_dir = first_level_dir / subject
        for run in runs:
            matrices.append(
                pd.read_csv(
                    subject_output_dir / f"sub-{subject}_run-{run}_design_matrix.csv",
                    index_col=0,
                )
            )
    return matrices


def concatenate_run_designs(
    run1: pd.DataFrame,
    run2: pd.DataFrame,
    shared_columns: list[str],
) -> pd.DataFrame:
    run1 = run1.copy()
    run2 = run2.copy()
    run1["run_regressor"] = 1

    for column in shared_columns:
        if column not in run1.columns:
            run1[column] = 0
        if column not in run2.columns:
            run2[column] = 0

    run1_unique = set(run1.columns) - set(run2.columns) - set(shared_columns)
    run2_unique = set(run2.columns) - set(run1.columns) - set(shared_columns)
    run1 = run1.rename(columns={column: f"run1_{column}" for column in run1_unique})
    run2 = run2.rename(columns={column: f"run2_{column}" for column in run2_unique})
    return pd.concat([run1, run2], ignore_index=True).fillna(0)


def fit_concatenated_first_level(
    config: AnalysisConfig,
    model_name: str,
    subjects: list[str],
    shared_columns: list[str],
    n_regressors_to_save: int,
    design_matrices: list[pd.DataFrame] | None = None,
    runs: tuple[int, int] = (1, 2),
) -> None:
    """Concatenate two runs per subject and save beta and z-score images."""

    first_level_dir = config.first_level_dir(model_name)
    first_level_dir.mkdir(parents=True, exist_ok=True)
    if design_matrices is None:
        design_matrices = load_design_matrices(first_level_dir, subjects, runs)

    design_index = 0
    for subject in subjects:
        print(f"running subject {subject}")
        subject_output_dir = first_level_dir / subject
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        concatenated_design = concatenate_run_designs(
            design_matrices[design_index],
            design_matrices[design_index + 1],
            shared_columns,
        )
        run_imgs = [
            nib.load(config.path(config.data_dir) / f"sub-{subject}_socialpred_run{run}.nii.gz")
            for run in runs
        ]
        concatenated_imgs = concat_imgs(run_imgs)

        concat_glm = FirstLevelModel(
            t_r=config.tr,
            noise_model="ar1",
            standardize=True,
            hrf_model=None,
            drift_model="cosine",
            high_pass=0.01,
            mask_img=config.gm_mask_img,
            smoothing_fwhm=6,
        )
        concat_glm.fit(concatenated_imgs, design_matrices=concatenated_design)

        plotting.plot_design_matrix(
            concat_glm.design_matrices_[0],
            output_file=subject_output_dir / "design_matrix_concatenated.png",
        )
        concatenated_design.to_csv(subject_output_dir / f"sub-{subject}_design_matrix_concatenated.csv")

        contrast_matrix = np.eye(concatenated_design.shape[1])
        for index in range(n_regressors_to_save):
            regressor = concatenated_design.columns[index]
            print(f"saving regressor for video {regressor}")
            effect = concat_glm.compute_contrast(contrast_matrix[index], output_type="effect_size")
            nib.save(effect, subject_output_dir / f"sub-{subject}_beta_video-{regressor}_gm_masked.nii.gz")

            z_score = concat_glm.compute_contrast(contrast_matrix[index], output_type="z_score")
            nib.save(z_score, subject_output_dir / f"sub-{subject}_z_score_video-{regressor}_gm_masked.nii.gz")

        design_index += len(runs)
