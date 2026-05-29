from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


DEFAULT_CONFOUNDS = [
    "csf",
    "white_matter",
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "framewise_displacement",
]


@dataclass
class AnalysisConfig:
    """Project paths and timing constants shared by analysis notebooks."""

    repo_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = Path("/Users/yiyuwang/Downloads/social_prediction_transformed_data_2mm")
    logfiles_dir: Path = Path("Data/logfiles")
    confounds_dir: Path = Path("Data/confounds")
    mask_dir: Path = Path("masks")
    figures_dir: Path = Path("figures")
    results_dir: Path = Path("Results")
    subjects_file: Path = Path("Data/included_SocialPred_subjects.csv")
    video_key_file: Path = Path(
        "/Users/yiyuwang/Dropbox/Projects/NEU_projects/SocialPrediction/Results/"
        "SocialPrediction_video_key.csv"
    )
    tr: float = 0.8
    n_tr: int = 675
    confounds_of_interest: list[str] = field(default_factory=lambda: DEFAULT_CONFOUNDS.copy())

    def path(self, value: str | Path) -> Path:
        value = Path(value)
        return value if value.is_absolute() else self.repo_root / value

    @property
    def gm_mask_img(self):
        import nibabel as nib

        return nib.load(self.path(self.mask_dir) / "gm_mask_icbm152_brain.nii.gz")

    def load_subjects(self) -> list[str]:
        subjects = pd.read_csv(self.path(self.subjects_file), header=None)
        return subjects[0].astype(str).tolist()

    def load_video_key(self) -> pd.DataFrame:
        return pd.read_csv(self.path(self.video_key_file))

    def first_level_dir(self, model_name: str) -> Path:
        return self.path(self.results_dir) / model_name / "1stLvl"

    def second_level_dir(self, model_name: str) -> Path:
        return self.path(self.results_dir) / model_name / "2ndLvl"
