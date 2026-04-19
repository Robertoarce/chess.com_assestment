# Chess Game Outcome Prediction

Author: Roberto Arce 
This project predicts the outcome of Chess.com Titled Tuesday blitz games as a three-class problem: white win, white loss, or draw. The core constraint across the repo is to use only pre-game information so the models do not leak the answer through post-game signals such as openings, PGN content, or result fields.

## Project Summary

The work is framed as a temporal, multiclass, and imbalanced prediction task. Players change over time, and draws are less frequent than wins or losses, so the project emphasizes evaluation choices that do not hide weak minority-class performance. The main model-selection metric is macro F1, with log loss used as the tie-breaker and recall and accuracy treated as secondary diagnostics.

The data pipeline starts from the Chess.com public API. The project notes show that tournament data is hierarchical, but the most useful game-level information comes from round groups and their games. That exploration was turned into a reusable data collection script in get_data.py, which fetches tournament data, engineers leakage-safe pre-game features, and writes the finished dataset used by the notebooks and the training script.

## Workflow

The repository follows a staged workflow:

1. API discovery and data extraction in API_discovery.ipynb and get_data.py.
2. Quality checks and exploratory analysis in EDA.ipynb.
3. Benchmarking and model selection in modeling.ipynb.
4. Standalone reproduction of the notebook pipeline in train_model.py.

The modeling strategy uses a time-aware split. The earlier tournament is used for training and validation, while the later tournament is kept as a final test set. This matches the real use case better than a random split and helps avoid leakage from mixing games from the same event across train and evaluation sets.

## Modeling Approach

The benchmark starts with a simple rule-based Elo baseline and then compares several standard tabular classifiers: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost. To address class imbalance, the project tests both class weighting and oversampling with ADASYN. The selection rule is validation-first: choose the model with the best validation macro F1, break ties with validation log loss, and use the test set only for confirmation rather than model picking.

Earlier exploratory files in the repo also document related experiments, including feature selection under higher-dimensional settings, multicollinearity checks, and imbalance handling through SMOTE or class weighting. Those comments support the final direction of keeping the benchmark practical, interpretable, and aligned with the assignment scope.

## Current Result

The current notebook and standalone script are aligned on model definitions, metric handling, and evaluation flow. The validation-selected model is XGB + ADASYN. Feature-importance analysis is used as an interpretation aid after model selection, while still keeping the overall approach focused on leakage-safe pre-game signals and validation-driven model choice.