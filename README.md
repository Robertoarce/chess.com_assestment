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


Modeling notebook file Objectives:
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



# OUTCOME

 chess git:(main) python train_model.py 
=== FETCHING DATA ===
 Saving finished dataframe into CSV...
 Fetching raw games data from API...
Saved 4063 rows to titled_tuesday_games_finished.csv
Data shape: (4063, 70)

=== DATA SUMMARY ===
Rows: 4063
Train/validation tournaments: ['Titled-Tuesday-Blitz-February-10-2026']
Held-out test tournament: Titled-Tuesday-Blitz-March-10-2026
Non-numeric features excluded: 9
Low-variance features dropped: 2
Dropped low-variance columns: ['group_index', 'game_is_rated']

=== SPLIT SUMMARY ===
Train:      1574 games (round_index < 9)
Validation: 451 games (round_index >= 9)
Test:       2038 games (Titled-Tuesday-Blitz-March-10-2026)
Train class distribution: {'win': 765, 'loss': 679, 'draw': 130}
Validation class distribution: {'win': 222, 'loss': 195, 'draw': 34}
Test class distribution: {'win': 964, 'loss': 923, 'draw': 151}

=== VALIDATION RESULTS ===
               name  accuracy  log_loss  macro_f1  weighted_f1  win_rec  loss_rec  draw_rec
       XGB + ADASYN    0.6075    0.8857    0.4494       0.5969   0.7207    0.5744    0.0588
      LR (balanced)    0.5543    1.0197    0.4424       0.5622   0.7432    0.3949    0.2353
        LR + ADASYN    0.6075    0.9796    0.4399       0.5830   0.8333    0.4462    0.0588
      RF (balanced)    0.5211    0.9902    0.4220       0.5372   0.7477    0.2974    0.3235
        RF + ADASYN    0.4900    0.9773    0.4171       0.5267   0.6486    0.3282    0.3824
     XGB (baseline)    0.6098    0.8983    0.4154       0.5800   0.8018    0.4974    0.0000
       GBM + ADASYN    0.5632    0.9296    0.4094       0.5604   0.7162    0.4821    0.0294
      LR (baseline)    0.6142    0.8925    0.4065       0.5697   0.8919    0.4051    0.0000
XGB (sample_weight)    0.5632    0.9153    0.4039       0.5483   0.7027    0.4974    0.0294
GBM (sample_weight)    0.5299    0.9915    0.3991       0.5382   0.6396    0.4872    0.0588
     GBM (baseline)    0.5588    0.9508    0.3955       0.5529   0.7523    0.4359    0.0000
      RF (baseline)    0.6031    0.8188    0.3940       0.5533   0.9054    0.3641    0.0000
     Rule-based Elo    0.2439    1.3168    0.2569       0.3029   0.2297    0.1590    0.8235

=== TEST RESULTS ===
               name  accuracy  log_loss  macro_f1  weighted_f1  win_rec  loss_rec  draw_rec
        RF + ADASYN    0.6624    0.8393    0.5243       0.6740   0.7811    0.6165    0.1854
      RF (balanced)    0.6516    0.8664    0.5137       0.6633   0.7739    0.6024    0.1722
        LR + ADASYN    0.6801    0.8545    0.5080       0.6717   0.7739    0.6815    0.0728
GBM (sample_weight)    0.6418    0.8785    0.5044       0.6572   0.7168    0.6436    0.1523
      LR (balanced)    0.6192    0.9187    0.5034       0.6485   0.6857    0.6143    0.2252
XGB (sample_weight)    0.6531    0.8392    0.5011       0.6548   0.7417    0.6490    0.1126
       XGB + ADASYN    0.6791    0.8027    0.4861       0.6647   0.7697    0.6923    0.0199
     GBM (baseline)    0.6771    0.7967    0.4856       0.6597   0.7998    0.6555    0.0265
     XGB (baseline)    0.6899    0.7895    0.4823       0.6655   0.7977    0.6891    0.0066
      LR (baseline)    0.6958    0.7546    0.4809       0.6682   0.8216    0.6782    0.0000
      RF (baseline)    0.6977    0.7297    0.4805       0.6679   0.8693    0.6327    0.0000
       GBM + ADASYN    0.6693    0.8014    0.4805       0.6577   0.7739    0.6663    0.0199
     Rule-based Elo    0.5226    0.9750    0.4802       0.5987   0.5124    0.5255    0.5695

=== SELECTED MODEL ===
Validation-selected model: XGB + ADASYN
Validation macro F1 / log loss: 0.4494 / 0.8857
Selected model on test macro F1 / log loss: 0.4861 / 0.8027
Best realized test model: RF + ADASYN

=== TOP PERMUTATION IMPORTANCES ===
                         feature  importance_mean  importance_std
                     rating_diff           0.0543          0.0129
                    rating_ratio           0.0367          0.0092
                  rating_diff_sq           0.0195          0.0112
   white_previous_draws_as_white           0.0194          0.0075
                    black_rating           0.0181          0.0086
  white_previous_draw_pct_as_any           0.0179          0.0125
black_previous_draw_pct_as_white           0.0178          0.0065
   white_previous_win_pct_as_any           0.0176          0.0072
                 rating_diff_abs           0.0167          0.0066
 white_previous_win_pct_as_black           0.0167          0.0107

=== TOP NATIVE FEATURE IMPORTANCES ===
                         feature  importance
           white_is_higher_rated      0.0765
  white_previous_points_as_white      0.0692
   white_previous_draws_as_white      0.0481
 white_previous_win_pct_as_white      0.0399
                    rating_ratio      0.0341
                     rating_diff      0.0341
  black_previous_points_as_white      0.0330
  white_previous_losses_as_white      0.0312
white_previous_loss_pct_as_white      0.0268
  black_previous_losses_as_black      0.0257
