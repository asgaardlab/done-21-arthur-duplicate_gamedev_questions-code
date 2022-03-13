import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils import paths, read, save, make_dir
from utils.models import get_X_y
from utils.models.scoring import multiple_k_scorer, predict_probabilities
from utils.consts import (
    datasets,
    n_procs,
    n_candidates,
    undersampling_percentages,
    best_candidates,
    best_undersampling,
)


def best_classifier(ds, c, p):
    """Reads the HP tuning results and returns the classifier
    that achieved the highest score
    """
    df = read(paths.cv_results(ds, c, p))
    # manual analysis suggests that rr@5 was a good metric
    # for choosing the best classifier
    params = df[df["rank_test_rr@5"] == 1].iloc[0].params

    # the dict objects in the dataframe
    # are sometimes converted to json format
    if type(params) == bytes:
        params = json.loads(params)

    rf = RandomForestClassifier(n_jobs=n_procs, **params)
    return rf


def train_best_classifier(ds, c, p):
    """Fits the best classifier from HP tuning on the train set
    for the given dataset and saves it
    """
    rf = best_classifier(ds, c, p)
    train = read(paths.train_set(ds, c, p))
    X, y = get_X_y(train)
    rf = rf.fit(X, y)

    make_dir(paths.classifiers_dir(ds))
    joblib.dump(rf, paths.classifier(ds, c, p))

    
def limit_candidates(df, c):
    df = df.groupby("dup_id").apply(
        lambda x: x.sort_values("score", ascending=False)[:c]
    )
    return df.reset_index(drop=True)

def score(ds, c, p, test_on=None):
    """Scores a given dataset by training on its train set and
    evaluating on the test set.
    Allows for using other datasets for evaluation using the test_on param
    """
    if test_on is None:
        dataset_test = ds
    else:
        dataset_test = test_on

    rf = joblib.load(paths.classifier(ds, c, p))

    test = read(paths.test_set(dataset_test))
    test = limit_candidates(test, c)
    X, y = get_X_y(test)

    scores = multiple_k_scorer(rf, X, y)

    scores["dataset_train"] = ds
    scores["candidates"] = c

    if test_on is not None:
        scores["dataset_test"] = test_on

    return scores


def candidate_performance(datasets, ns, p):
    """Tests every dataset on its own test set using
    different numbers of candidates and saves a summary dataset"""
    df = []
    for ds in datasets:
        print(f"Evaluating classifiers on different candidates for {ds}")
        for n in ns:
            print(f"- {n} candidates")
            df.append(score(ds, n, p))

    df = pd.DataFrame(df)
    save(df, paths.candidates_evaluation())


def cross_dataset_performance(datasets, n, p):
    """Evaluates the performance of classifiers trained on one dataset in using
    the others as evaluation"""
    df = []

    print(f"Evaluating classifiers cross-datasets")

    for ds1 in datasets:
        for ds2 in datasets:
            print(f"- Training on: {ds1}; testing on: {ds2}")
            df.append(score(ds1, n, p, test_on=ds2))
    
    df = pd.DataFrame(df)
    df = df[["dataset_train", "dataset_test", "rr@5", "rr@10", "rr@20"]]
    df = df.sort_values('dataset_test')

    save(df, paths.cross_dataset_performance())
    
def misclassified_dups(ds, c, p):
    """Makes a dataset of the duplicate questions that do not have their main question
    in the 20 top ranked pairs for the dataset"""
    def select_misclassified(df, dups):
        """Selects the true pair and the top ranked pair for each misclassified duplicate"""
        dup = df.name
        df = df.sort_values('pred', ascending=False)
        if df[:20]['is_dup'].any():
            # correctly classified = nothing to return
            res = pd.DataFrame()
        else:
            # if the dup is in the list of candidates at all
            has_dup = df.is_dup.any()
            
            if has_dup:
                true_dup = df[df.is_dup].iloc[0]['candidate_id']
            else:
                true_dup = dups[dups.dup_id == dup].iloc[0]['main_id']
                
            top_ranked = df[~df.is_dup].iloc[0]['candidate_id']
            
            res = pd.DataFrame([{'main_id': true_dup, 'top_ranked': top_ranked, 'has_dup': has_dup}])
        return res

    print(f'- Selecting misclassified duplicates for {ds}')
        
    rf = joblib.load(paths.classifier(ds, c, p))
    
    test = read(paths.test_set(ds))
    test = limit_candidates(test, c)
    X, y = get_X_y(test)
    
    test = predict_probabilities(rf, X, y)
    
    dup_pairs = read(paths.dup_pairs(ds))
    
    missed = test.groupby('dup_id').apply(lambda df: select_misclassified(df, dup_pairs))

    if len(missed) > 0:
        missed = missed.reset_index()
        missed['has_dup'] = missed.has_dup.apply(bool)
        
        missed = missed.drop(columns=['level_1'])
        
        # adds URLs for easier analysis
        if ds == 'gamedev_se':
            to_url = lambda i: f'https://gamedev.stackexchange.com/questions/{i}/'
        else:
            to_url = lambda i: f'https://stackoverflow.com/questions/{i}/'
        
        for c in [c for c in missed.columns if c != 'has_dup']:
            missed[c + '_url'] = missed[c].apply(to_url)
        
        if 'gamedev' not in ds:
            ds = 'so_sample'
        
    save(missed, paths.misclassified_duplicates(ds))


def train_classifiers(ds, ns, ps):
    """Trains classifiers for all combinations of candidates and undersampling
    percentages
    """
    print(f"Training classifiers for {ds}.")
    for n in ns:
        for p in ps:
            print(f"- Training classifiers for {n} candidates")
            train_best_classifier(ds, n, p)


def main(datasets, ns, ps, best_n, best_p):
    for ds in datasets:
        train_classifiers(ds, ns, ps)

    for ds in ['gamedev_se', 'gamedev_so', 'so_samples/sample_0']:
        misclassified_dups(ds, best_n, best_p)

    candidate_performance(datasets, ns, best_p)
    cross_dataset_performance(datasets, best_n, best_p)


if __name__ == "__main__":
    main(
        datasets,
        n_candidates,
        undersampling_percentages,
        best_candidates,
        best_undersampling,
    )
