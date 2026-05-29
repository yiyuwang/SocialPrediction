from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp

from agentprediction.events import LOGFILE_HEADERS, get_condition, get_subjective_prior


def parse_behavior_lines(lines, video_key: pd.DataFrame, headers=LOGFILE_HEADERS):
    header_index = {name: idx for idx, name in enumerate(headers)}
    for trial_num, line in enumerate(lines):
        cols = line.split()
        video_number = int(cols[header_index["video_number"]])
        trial_condition = get_condition(int(cols[header_index["trial_condition"]]))
        run_condition = get_condition(int(cols[header_index["run_condition"]]))
        prediction = int(abs(float(cols[header_index["prediction"]])))
        _, prediction_condition = get_subjective_prior(prediction, video_number, video_key)
        run = int(cols[header_index["run_number"]])
        surprise = float(cols[header_index["surprise"]])

        if prediction_condition == "Pattern":
            fb_trial_type = "fb_Pattern_Congruent" if trial_condition == "Pattern" else "fb_Pattern_PE"
        elif prediction_condition == "Social":
            fb_trial_type = "fb_Social_Congruent" if trial_condition == "Social" else "fb_Social_PE"
        else:
            fb_trial_type = "fb_Neither"

        yield [trial_num, trial_condition, prediction_condition, run, run_condition, surprise, fb_trial_type]


def create_behavior_dataframe(task_file: str | Path, run: int, video_key: pd.DataFrame) -> pd.DataFrame:
    with open(task_file) as handle:
        rows = list(parse_behavior_lines(handle.readlines(), video_key))
    df = pd.DataFrame(
        rows,
        columns=[
            "trial_num",
            "feedback_condition",
            "prediction_condition",
            "run",
            "run_condition",
            "surprise",
            "fb_trial_type",
        ],
    )
    return df[df["run"] == run].reset_index(drop=True)


def find_task_file(logfiles_dir: str | Path, subject: str) -> Path:
    matches = sorted(Path(logfiles_dir).glob(f"*{subject}*edited.txt"))
    if not matches:
        raise FileNotFoundError(f"No edited logfile found for subject {subject}")
    return matches[0]


def build_sliding_window_dataframe(
    subjects: list[str],
    logfiles_dir: str | Path,
    video_key: pd.DataFrame,
    window: int,
    runs=(1, 2),
) -> pd.DataFrame:
    all_rows = []
    for subject in subjects:
        task_file = find_task_file(logfiles_dir, subject)
        for run in runs:
            df = create_behavior_dataframe(task_file, run, video_key)
            run_condition = df.loc[0, "run_condition"]
            rows = []
            for trial in range(window, len(df)):
                prior = df.iloc[trial - window : trial]
                prediction_condition = df.loc[trial, "prediction_condition"]
                prediction = int(prediction_condition == run_condition)
                prior_feedback_prob = np.mean(prior["feedback_condition"].to_numpy() == run_condition)
                prior_prediction_prob = np.mean(prior["prediction_condition"].to_numpy() == run_condition)
                rows.append(
                    {
                        "sub": subject,
                        "run": run,
                        "run_condition": run_condition,
                        "trial_num": trial,
                        "probability": prior_feedback_prob,
                        "difference": prediction - prior_feedback_prob,
                        "prediction": prediction,
                        "prediction_prob": prior_prediction_prob,
                        "prediction_condition": prediction_condition,
                        "surprise": df.loc[trial, "surprise"],
                        "congruent": df.loc[trial, "fb_trial_type"],
                    }
                )
            all_rows.append(pd.DataFrame(rows))
    return pd.concat(all_rows, ignore_index=True)


def fit_sliding_window_logits(
    subjects: list[str],
    logfiles_dir: str | Path,
    video_key: pd.DataFrame,
    windows=range(1, 11),
) -> tuple[pd.DataFrame, dict[int, object]]:
    rows = []
    models = {}
    for window in windows:
        df = build_sliding_window_dataframe(subjects, logfiles_dir, video_key, window)
        model_df = df.dropna(subset=["prediction", "probability", "sub"])
        result = smf.logit("prediction ~ probability", data=model_df).fit()
        models[window] = result
        rows.append(
            {
                "wmax": window,
                "condition": "all",
                "pvalues": result.pvalues.iloc[1],
                "betas": result.params.iloc[1],
                "se": result.bse.iloc[1],
                "aic": result.aic,
                "bic": result.bic,
            }
        )
    return pd.DataFrame(rows), models


def plot_sliding_window_betas(res_df: pd.DataFrame):
    x = res_df["wmax"]
    y = res_df["betas"]
    err = res_df["se"]

    plt.figure(figsize=(7, 6))
    plt.errorbar(x, y, yerr=err, fmt="o-", color="gray", capsize=4, ecolor="gray", elinewidth=3, markersize=8, linewidth=3)
    for i, p_value in enumerate(res_df["pvalues"]):
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        if stars:
            plt.text(x.iloc[i], y.iloc[i] + err.iloc[i] + 0.1, stars, fontsize=25, ha="center", color="black")

    plt.xlabel("sliding window (trials)", fontsize=24, fontname="Arial")
    plt.ylabel("Logistic Regression Coefficient", fontsize=24, fontname="Arial")
    plt.xticks(x, fontsize=20, fontname="Arial")
    plt.yticks(fontsize=20, fontname="Arial")
    plt.ylim(0, 2.5)
    plt.title(" ")
    plt.tight_layout()
    plt.show()


def label_congruence(row):
    if "Congruent" in row["congruent"]:
        return "congruent"
    if "PE" in row["congruent"]:
        return "incongruent"
    return "neither"


def prepare_congruence_dataframe(df: pd.DataFrame, prediction_condition: str | None = None) -> pd.DataFrame:
    reg_df = df.copy()
    if prediction_condition is not None:
        reg_df = reg_df[reg_df["prediction_condition"] == prediction_condition].reset_index(drop=True)
    reg_df["congruence"] = reg_df.apply(label_congruence, axis=1)
    reg_df = reg_df[reg_df["congruence"] != "neither"].reset_index(drop=True)
    reg_df["congruence_coded"] = reg_df["congruence"].map({"congruent": 0, "incongruent": 1})
    return reg_df


def paired_surprise_test(reg_df: pd.DataFrame):
    subj_means = reg_df.groupby(["sub", "congruence"])["surprise"].mean().unstack("congruence")
    diff = (subj_means["incongruent"] - subj_means["congruent"]).dropna()
    t_stat, p_value = ttest_1samp(diff, popmean=0)
    print("t =", t_stat, "p =", p_value)
    print("Mean(PE - congruent) =", diff.mean())
    return t_stat, p_value, diff


def congruence_wide(reg_df: pd.DataFrame, incongruent_label: str = "Incongruent") -> pd.DataFrame:
    wide = reg_df.groupby(["sub", "congruence"])["surprise"].mean().reset_index()
    wide["congruence"] = wide["congruence"].apply(lambda x: incongruent_label if x == "incongruent" else "Congruent")
    return wide.pivot(index="sub", columns="congruence", values="surprise").reset_index().dropna()
