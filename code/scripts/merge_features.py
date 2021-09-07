import pandas as pd
import numpy as np

from utils import paths, read, save
from utils.consts import features, datasets, n_candidates, undersampling_percentages


def check_features(feats, cands):
    """Checks if the candidates dataframe is the same as the features
    dataframe after merging (e.g., if there are no repeated entries)
    """
    same_size = len(cands) == len(feats)
    same_dups = set(cands.dup_id) == set(feats.dup_id)
    same_rels = set(cands.candidate_id) == set(feats.candidate_id)

    return same_size and same_dups and same_rels


def make_test(ds, feats):
    """Creates a test dataset by joining features to the test candidate pairs"""
    cands = read(paths.test_candidate_pairs(ds))
    test = merge_features(ds, cands.copy(), feats)

    assert check_features(
        test, cands
    ), "The test dataset is different from its candidates!"
    assert not test.isna().any().any(), "The test dataset has NaNs!"

    save(test, paths.test_set(ds))


def make_train(ds, n, p, feats):
    """Creates a train dataset by joining features to one of the train candidate pairs set"""
    cands = read(paths.train_candidate_pairs(ds, n, p))
    train = merge_features(ds, cands.copy(), feats)

    assert check_features(
        train, cands
    ), f"The train dataset ({n}, {p}) is different from its candidates!"
    assert not train.isna().any().any(), f"The train dataset ({n}, {p}) has NaNs!"

    save(train, paths.train_set(ds, n, p))


def merge_features(ds, cands, feats):
    """Merges the features with the candidate pairs for a give dataset"""
    # only keeps the relevant columns for the feature set
    # score will later be used to truncate the test set and allow for
    # evaluation on a different number of candidates
    cands = cands[["dup_id", "candidate_id", "score", "is_dup"]]
    for f in feats:
        df_feat = read(paths.feature(ds, f))
        cands = cands.merge(df_feat, on=["candidate_id", "dup_id"], how="left")
    return cands


def make_train_multi(ds, ns, ps, feats):
    """Creates train sets of features for all combinations of candidate pairs and
    undersampling percentages
    """
    for n in ns:
        for p in ps:
            print(f"- Making train set ({n}, {p})")
            make_train(ds, n, p, feats)


def make_sets_multi(datasets, ns, ps, feats):
    """Creates train and test sets of features for all combinations of candidate pairs and
    undersampling percentages for the given datasets
    """
    for ds in datasets:
        print(f"Creating train and test sets for {ds}")
        make_train_multi(ds, ns, ps, feats)
        make_test(ds, feats)


def make_sets(datasets, n, p, feats):
    """Creates train and test sets of features for one value of candidate pairs and
    undersampling percentages for the given datasets
    """
    for ds in datasets:
        print(f"Creating train and test sets for {ds}")
        make_train(ds, n, p, feats)
        make_test(ds, feats)


if __name__ == "__main__":
    make_sets_multi(datasets, n_candidates, undersampling_percentages, features)
