# AI/ML Team Take Home Exercise: Predicting Chess Game Outcomes

**Author:** Roberto Arce

## How To Run

This project can be run either through the standalone scripts or through the notebooks.

### Setup

Create and activate the environment, then install the dependencies:

Using Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Main Commands

Run the full benchmark outside the notebook:

```bash
python train_model.py
```

This is the main script to run. It also fetches the tournament data and rebuilds the finished dataset through the data pipeline before training the models.

Notebooks available:

The Notebooks are the most useful for the project information:

1. `modeling.ipynb`: the main benchmark notebook, including the split logic, model comparison, plots, feature importance, and final recommendation.
2. `EDA.ipynb`: data checks and exploratory analysis, useful to understand distributions, quality checks, and feature behavior.
3. `API_discovery.ipynb`: notes on the Chess.com API structure and what information was taken from tournaments, rounds, groups, and games, with some data exploration and quality verification.

---

# How the Problem was framed:
1. This is a temporal multiclass and imbalanced prediction problem:
    1. Players evolve over time.
    1. Draw has much less volume than the others.
1. The objective of the model is to predict * with equal importance* white's win loose or draw outcome.
    1. Thus main metric will be macro F1 (to give equal weight to each class)
    1. Followed by log loss, recall and accuracy as secondary metrics.

About the data: 

1. Since we are going to split the data for train and evaluation and this is a temporal model (people get better/worse with time) we need to separate the data. For this we will use one (early) tournament for train and split it into train and validation and use the remaining tournament as the one for test.
1. There has been already a lot of feature engineering work before, thus we will need to trim/regulate the features.

About the modeling: 

1. Our Baseline will be a very vanila model base on Elo difference AND we will give ficticious probabilities (to avoid over engineering).
1. Since there is an imbalance in data we will use 2 balancing techniques (ADASYN and model class Weights).
1. The main selected models to test will be LR (excelent for later interpretability), tree base model (Random Forest, GradientBoostingClassifier), and the best in class XGBoost that is a strong tabular model baseline. All of them are much easy to interpret compared to DL models.
1. Finally this notebook will be served for the final script as required by the assignment.


Modeling Objectives:
1. Create a benchmark playground for different models and strategies, to with different:
    1. Models (No hyperparameterization with optuna)
    1. Pipelines (only scikit learn pipelines)
    1. Strategies (model alone, layers, ensemble) <-- more possible but this is good enough for the timing
    1. Features selections
1. Regarding the model selection: this is based on the macro F1 + log loss (if there is a tie) on validation data; the test set is only for confirmation (not selection).


## Project  Workflow

The repository follows a staged workflow:

1. API discovery and data extraction in API_discovery.ipynb and get_data.py.
2. Quality checks and exploratory analysis in EDA.ipynb.
3. Benchmarking and model selection in modeling.ipynb.
4. Standalone reproduction of the notebook pipeline in train_model.py.

Please note that there are indepth comments in the notebooks!!! 

## Modeling Approach

The benchmark starts with a simple rule-based Elo baseline and then compares several standard tabular classifiers: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost. To address class imbalance, the project tests both class weighting and oversampling with ADASYN. 

The selection rule is validation-first: choose the model with the best validation macro F1 and log loss for tied models; then use the test set only for confirmation rather than model picking.

There is a lot of exploration in the notebooks such as: experiments, feature selection, multicollinearity checks, and imbalance handling through ADSYN or class weighting. 

## Result

The validation-selected model is XGB + ADASYN. Feature-importance analysis is used as an interpretation aid after model selection, while still keeping the overall approach focused on leakage-safe pre-game signals (this by using the round split) and validation-driven model choice.

## Overall features importance results:
The best model strenght is still base on the rating, but its improvement over the rule-based baseline comes from combining that signal with past performance and draw-related history.
> more details at title:  'Note on Final Feature Importance Results' (in the modeling notebook)

## Final words on the Model Selected

The final model selected on the validation split is `XGB + ADASYN`.

Reasons:

1. Best validation macro F1 among the benchmarked models.
2. It treats `win`, `loss`, and `draw` more evenly.
3. ADASYN helps the model pay more attention to the minority draw class.

But the test results differ, the RF + ADASYN is better on the held-out test set:

1. Yes, but the test split is used only as final confirmation, not for model selection.
2. If test performance differs from validation performance, that is part of the evaluation story rather than a reason to change the selected model after the fact.
3. The generalization did not  generalize best to future data; given the 2 issues (temporal split and class imbalance)
4. The final recommendation is therefore based on validation.


# What was avoided or could be implemented in future:

1. The usage of player stats and profiles, to avoid more time usage and complexity.
1. A more robust time-based validation scheme with more than one temporal validation window.
1. Add more data (tournaments) to have a broader range (this will most likely also improve variance and confidence of the model).
1. A two staged ensemble approach, by using one model for draw-win/loss classification and then win-loss classification.


# About the usage of AI:

Every cell in the notebooks that has been generated by IA is explicitly marked.
I focus on non-valuable tasks, such as creating visualizations and fixing syntax (when needed). 
In most cases i added the initial prompt (most of the times the initial prompt was not enough but the idea of the work stayed the same)
I also asked the translation from notebook to file, which also required cleaning afterwards.

Most used LLM (by order): code: ChatGPT, Viz: Claude + lechat.

# Final words from the Author

This has been a really passionate test!. As a chess player I was sure that elo would be the most strong signal, (and this was also found during my initial models where a model with just a simple elo difference rule was able to predict quite reasonably well),but then after (and confirmed by the feature importance test) history features added a visible improvement. Certainly there is more to investigate!
