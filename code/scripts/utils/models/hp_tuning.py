import numpy as np
from . import get_X_y
from ..consts import search_n_iters
from .scoring import multiple_k_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def get_folds(df, n):
    """Creates custom folds to be used in hyperparameter tuning
    These folds guarantee that no test dups are contained in the train
    folds

    If we use sklearn's default algorithm, it would randomly select pairs
    and the test dups have pairs in the train set, which could lead to
    leakage
    """
    dups = df[df.is_dup][["dup_id"]]
    dups = dups.drop_duplicates()

    # shuffle dups
    dups = dups.sample(frac=1, random_state=42)
    dups = dups.dup_id

    folds = []
    for test_dups in np.array_split(dups, n):
        test_ids = df[df.dup_id.isin(test_dups)]
        test_ids = test_ids.index

        train_ids = df[~df.dup_id.isin(test_dups)]
        train_ids = train_ids.index

        folds.append((train_ids, test_ids))
    return folds


def random_forest_tuner(folds):
    """Returns an instance of a RandomizedSearchCV with pre-defined
    parameters for a random forest
    """
    # parameters used for tuning random forests
    randomforest_hp_grid = {
        "n_estimators": [int(x) for x in np.linspace(50, 1000, num=50)],
        "max_depth": [int(x) for x in np.linspace(5, 100, num=50)],
        "min_samples_split": [2, 3, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "random_state": [42],
    }

    est = RandomForestClassifier()
    return RandomizedSearchCV(
        est,
        randomforest_hp_grid,
        n_iter=search_n_iters,
        cv=folds,
        scoring=multiple_k_scorer,
        verbose=2,
        random_state=42,
        n_jobs=7,
        refit=False,
    )


def tune_train_set(train, n):
    """Tunes hyperparameters using the provided train set"""
    X, y = get_X_y(train)
    folds = get_folds(train, n)
    rs = random_forest_tuner(folds)

    return rs.fit(X, y)
