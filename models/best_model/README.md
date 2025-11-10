# Best model artifact

This directory captures metadata about the best performing configuration discovered during model selection. The fine-tuned head parameters are stored in `head_state_dict.json` and are intended to be loaded on top of the Hugging Face base model identified in `config.json`. The example head uses a reduced dimensionality so that the artifact remains lightweight in version control. The accompanying `metrics.json` mirrors the summary produced by the experiment notebook so downstream automation can trigger deployments or alerts.

> **Note:** The lightweight head weights stored here are illustrative. Re-run `notebooks/model_selection.ipynb` to regenerate production-ready weights and overwrite this placeholder.
