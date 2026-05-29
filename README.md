# The Neural Architecture of Predicting Agentic Behavior

Code for the analysis in:

**Yiyu Wang, Juliet Y. Davidow, Richard D. Lane, Ajay B. Satpute**  
*The Neural Architecture of Predicting Agentic Behavior: Dissociable Systems for Mental Model Formation and Updating*

Preprint: [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.10.658968v1)  
Stimulus repository: [AgentPredictionTask](https://github.com/yiyuwang/AgentPredictionTask)

## Repository Structure

```text
.
├── preprocessing/                 # Cluster preprocessing scripts
├── src/agentprediction/            # Shared Python analysis helpers
├── masks/                          # Public masks and templates used by notebooks
├── 2_FirstLevel_...ipynb           # Numbered analysis notebooks
├── 3_ANOVA_...ipynb
├── ...
├── requirements.txt
└── pyproject.toml
```

Large/private analysis inputs and outputs are intentionally ignored by git:

- `Data/`
- `Results/`
- `figures/`
- `Figures/`
- `Manuscripts/`
- `InDevCode/`

## Setup

The notebooks expect the private data folders listed above to exist locally. The default paths live in `src/agentprediction/config.py`; edit `AnalysisConfig` in a notebook if your transformed fMRI data or video-key file live somewhere else.

## Workflow

Run the preprocessing scripts first, then run the numbered notebooks in order. The notebooks remain the main place for figures and interactive inspection; repeated code now lives in `src/agentprediction`.

1. `preprocessing/`
   - `wrapper_run_fmriprep_fmapreg.sh`
   - `edit_fmap_json.py`
   - `run_fmriprep_fmap_reg.sh`
   - `organize_confounds_cluster.sh`
   - `apply_transform_SP_cluster.sh`
2. `2_FirstLevel_Models_subjective_prior_ConcatRuns.ipynb`
3. `3_ANOVA_KMeans_knee.ipynb`
4. `4_Behavioral_Check.ipynb`
5. `5_FirstLevel_Models_VideoOn_ConcatRuns.ipynb`
6. `6_SecondLevel_Models_VideoOn.ipynb`
7. `7_FirstLevel_Models_subjective_prior_ConcatRuns_cue_offset.ipynb`
8. `8_ANOVA_KMeans_cue_offset.ipynb`
9. `9_Split_half_A.ipynb`
10. `10_Split_half_split_B.ipynb`

## Shared Code

The reusable package is organized by analysis responsibility:

- `config.py`: project paths, timing constants, subjects, masks, and result directories.
- `events.py`: logfile headers, condition labels, subjective-prior scoring, and event dataframe creation.
- `confounds.py`: fMRIPrep confound loading and outlier-column handling.
- `first_level.py`: per-run design matrix fitting and concatenated first-level GLMs.
- `second_level.py`: video-on group model helpers and thresholding.
- `anova.py`: omnibus ANOVA, cluster summaries, and shared ANOVA plotting helpers.
- `behavior.py`: behavioral logfile parsing, sliding-window models, and surprise-rating summaries.
- `plotting.py`: common clustering and plotting utilities used by exploratory notebooks.

When adding new analysis variants, keep the notebook focused on analysis-specific parameters and figures, and put reusable parsing/modeling helpers in `src/agentprediction`.

## Citation

If you use these stimuli or analyses in your work, please cite:

```bibtex
@article{wang2025disentangling,
  title={Disentangling Prediction and Feedback in Social Brain Networks: A Predictive Processing Approach},
  author={Wang, Yiyu and Davidow, Juliet Y and Lane, Richard D and Satpute, Ajay B},
  journal={bioRxiv},
  pages={2025--06},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

Questions: a.satpute@northeastern.edu or yiyuwang@stanford.edu
