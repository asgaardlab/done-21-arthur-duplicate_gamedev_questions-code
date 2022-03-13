import numpy as np
import pandas as pd

from utils import paths, read, save
from utils.models.hp_tuning import tune_train_set
from utils.consts import datasets, n_candidates, undersampling_percentages


def random_forest_search(ds, c, p):
    """Tunes hyperparameters for the train set with c candidates and p undersampling percentage
    for the given dataset using the provided hyperparameter tuning parameters
    """
    train = read(paths.train_set(ds, c, p))
    results = tune_train_set(train, 5)
    results = pd.DataFrame(results.cv_results_)
    save(results, paths.cv_results(ds, c, p))


def tune_multiple_sets(datasets, ns, ps):
    for ds in datasets:
        print(f"Tuning hyperparameters for {ds}")
        for p in ps:
            for n in ns:
                print(
                    f"- Tuning HPs for the train set with {n} candidates, {p} percent dups"
                )
                rf = random_forest_search(ds, n, p)


if __name__ == "__main__":
    tune_multiple_sets(datasets, n_candidates, undersampling_percentages)
