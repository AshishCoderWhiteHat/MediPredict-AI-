# Processed Data Directory
This directory stores ephemeral processed splits, cleaned intermediate CSVs, or validation batches.
All model transformations are executed through Scikit-Learn Pipelines fitted strictly on training folds to prevent data leakage.
