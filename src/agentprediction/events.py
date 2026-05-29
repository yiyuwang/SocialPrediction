from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


LOGFILE_HEADERS = np.array(
    [
        "obs_video_name",
        "fd_video_name",
        "video_number",
        "trial_condition",
        "run_number",
        "run_condition",
        "obs_video_onset",
        "obs_video_offset",
        "obs_video_duration_method1",
        "obs_video_duration_method2",
        "prediction",
        "prediction_x",
        "prediction_y",
        "prediction_RT",
        "prediction_onset",
        "fb_video_onset",
        "fb_video_offset",
        "fb_video_duration_method1",
        "fb_video_duration_method2",
        "surprise",
        "surprise_RT",
        "surprise_onset",
    ]
)


def _column_index(headers: np.ndarray, name: str) -> int:
    return int(np.where(headers == name)[0][0])


def get_condition(condition_number: int) -> str:
    labels = {1: "Pattern", 2: "Social"}
    if condition_number not in labels:
        raise ValueError(f"Unknown condition number: {condition_number}")
    return labels[condition_number]


def get_condition_number(condition: str) -> int:
    labels = {"Pattern": 1, "Social": 2}
    if condition not in labels:
        raise ValueError(f"Unknown condition label: {condition}")
    return labels[condition]


def get_subjective_prior(prediction: int, video_number: int, video_key: pd.DataFrame) -> tuple[int, str]:
    row = video_key.loc[video_key.vid_num == video_number].iloc[0]
    social_prior = row["Social_correct"]
    pattern_prior = row["Pattern_correct"]

    special_pattern_answers = {
        7: {4, 2},
        8: {3, 2},
        6: {4, 1},
    }
    if pattern_prior in special_pattern_answers:
        if prediction in special_pattern_answers[pattern_prior]:
            return 1, "Pattern"
        if prediction == social_prior:
            return 2, "Social"
        return 0, "Neither"

    if prediction == social_prior:
        return 2, "Social"
    if prediction == pattern_prior:
        return 1, "Pattern"
    return 0, "Neither"


def _feedback_trial_type(subjective_prior_condition: str, trial_condition: str) -> str:
    if subjective_prior_condition == "Pattern":
        return "fb_Pattern_Congruent" if trial_condition == "Pattern" else "fb_Pattern_PE"
    if subjective_prior_condition == "Social":
        return "fb_Social_Congruent" if trial_condition == "Social" else "fb_Social_PE"
    if subjective_prior_condition == "Neither":
        return "fb_Neither"
    raise ValueError(f"Unknown subjective prior condition: {subjective_prior_condition}")


def parse_subjective_prior_events(
    lines: Iterable[str],
    video_key: pd.DataFrame,
    include_cue_offset: bool = False,
    headers: np.ndarray = LOGFILE_HEADERS,
):
    """Yield first-level events for the subjective-prior models."""

    for line in lines:
        cols = line.split()
        video_number = int(cols[_column_index(headers, "video_number")])
        trial_condition = get_condition(int(cols[_column_index(headers, "trial_condition")]))
        prediction = int(abs(float(cols[_column_index(headers, "prediction")])))

        _, prior_condition = get_subjective_prior(prediction, video_number, video_key)
        obs_trial_type = f"obs_{prior_condition}"
        fb_trial_type = _feedback_trial_type(prior_condition, trial_condition)

        video_onset = float(cols[_column_index(headers, "obs_video_onset")])
        video_offset = float(cols[_column_index(headers, "obs_video_offset")])
        fb_video_onset = float(cols[_column_index(headers, "fb_video_onset")])
        fb_video_offset = float(cols[_column_index(headers, "fb_video_offset")])
        run = int(cols[_column_index(headers, "run_number")])
        prediction_onset = float(cols[_column_index(headers, "prediction_onset")])
        surprise_onset = float(cols[_column_index(headers, "surprise_onset")])

        yield [video_onset, video_offset - video_onset, obs_trial_type, run]
        if include_cue_offset:
            yield [video_offset, 1, "cue_offset", run]
        yield [fb_video_onset, fb_video_offset - fb_video_onset, fb_trial_type, run]
        yield [surprise_onset, 1, "surprise_onset", run]
        yield [prediction_onset, 1, "pred_onset", run]


def parse_video_on_events(
    lines: list[str],
    n_tr: int = 675,
    tr: float = 0.8,
    headers: np.ndarray = LOGFILE_HEADERS,
):
    """Yield cue, feedback, rest, and rating events for the video-on model."""

    for index, line in enumerate(lines):
        cols = line.split()
        video_onset = float(cols[_column_index(headers, "obs_video_onset")])
        video_offset = float(cols[_column_index(headers, "obs_video_offset")])
        fb_video_onset = float(cols[_column_index(headers, "fb_video_onset")])
        fb_video_offset = float(cols[_column_index(headers, "fb_video_offset")])
        run = int(cols[_column_index(headers, "run_number")])
        prediction_onset = float(cols[_column_index(headers, "prediction_onset")])
        surprise_onset = float(cols[_column_index(headers, "surprise_onset")])

        rest_onset1 = prediction_onset + 4
        rest_onset2 = surprise_onset + 4
        if index in {19, 39}:
            next_line_onset = n_tr * tr
        else:
            next_line_onset = float(lines[index + 1].split()[_column_index(headers, "obs_video_onset")])

        yield [video_onset, video_offset - video_onset, "cue_VideoOn", run]
        yield [fb_video_onset, fb_video_offset - fb_video_onset, "fb_VideoOn", run]
        yield [rest_onset1, fb_video_onset - rest_onset1, "rest", run]
        yield [rest_onset2, next_line_onset - rest_onset2, "rest", run]
        yield [surprise_onset, 1, "surprise_onset", run]
        yield [prediction_onset, 1, "pred_onset", run]


def create_events_dataframe(
    task_file: str | Path,
    run: int,
    parser,
    **parser_kwargs,
) -> pd.DataFrame:
    with open(task_file) as task_csv_file:
        events = list(parser(task_csv_file.readlines(), **parser_kwargs))

    df = pd.DataFrame(events, columns=["onset", "duration", "trial_type", "run"])
    return df.loc[df["run"] == run].drop(columns=["run"])
