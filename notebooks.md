# Notebook Workflow

Run the notebooks in numerical order after preprocessing.

1. `preprocessing/`: fMRIPrep, confound organization, and transforms.
2. `2_FirstLevel_Models_subjective_prior_ConcatRuns.ipynb`: first-level subjective-prior GLM.
3. `3_ANOVA_KMeans_knee.ipynb`: omnibus ANOVA and clustering.
4. `4_Behavioral_Check.ipynb`: behavioral summaries and checks.
5. `5_FirstLevel_Models_VideoOn_ConcatRuns.ipynb`: first-level video-on GLM.
6. `6_SecondLevel_Models_VideoOn.ipynb`: group-level video-on contrasts.
7. `7_FirstLevel_Models_subjective_prior_ConcatRuns_cue_offset.ipynb`: cue-offset first-level model.
8. `8_ANOVA_KMeans_cue_offset.ipynb`: cue-offset ANOVA and clustering.
9. `9_Split_half_A.ipynb`: split-half analysis A.
10. `10_Split_half_split_B.ipynb`: split-half analysis B.

Shared code now lives in `src/agentprediction`. Keep figure-heavy exploratory plotting
inside notebooks, and move repeated loading, event parsing, GLM, or clustering helpers into
the package.
