import numpy as np
import pandas as pd
from multiprocessing import Pool

from utils import paths, read, save, QuestionComp
from utils.consts import datasets, n_procs, features, text_columns


def save_features(df, ds, feature_name, proc_num):
    """Saves only the relevant columns for a set of features"""

    def cols_to_save(df):
        return ["dup_id", "candidate_id"] + [c for c in df.columns if "_sim" in c]

    save_path = paths.feature(ds, feature_name, proc_num)
    cols = cols_to_save(df)
    save(df[cols], save_path)


def calc_feature(ds, feature, cols, proc_num=None):
    """Calculates the feature values (similarity scores) for candidates from a given dataset
    proc_num serves to select a chunk of candidates as opposed to all of them
    """

    def compare_pairs(df, qc, f, c):
        dup_id, dup_index = df.name
        cand_indexes = df["candidate_corpus_index"]
        df[f"{c}_{f}_sim"] = qc.compare(dup_index, cand_indexes)
        return df

    candidates = read(paths.candidate_pairs(ds, proc_num))

    for c in cols:
        qc = QuestionComp(ds, feature, c)
        f = lambda df: compare_pairs(df, qc, feature, c)
        candidates = candidates.groupby(["dup_id", "dup_corpus_index"]).apply(f)
        candidates = candidates.reset_index(drop=True)

    save_features(candidates, ds, feature, proc_num)


def split_candidates(ds, n_procs):
    """Splits candidates into chunks to allow for multiprocessing
    during feature calculation
    """
    candidates = read(paths.candidate_pairs(ds))

    dups = candidates["dup_id"].unique()
    np.random.shuffle(dups)
    split_dups = np.array_split(dups, n_procs)

    for i, c in enumerate(split_dups):
        df = candidates[candidates.dup_id.isin(c)]
        save(df, paths.candidate_pairs(ds, i))


def merge_datasets(ds, n_procs, feats):
    """Merges the chunks of feature dataframes into a single dataframe
    for each feature
    """
    for f in feats:
        # merge and save
        fs = [read(paths.feature(ds, f, i)) for i in range(n_procs)]
        df_feat = pd.concat(fs).reset_index(drop=True)
        save(df_feat, paths.feature(ds, f))

        # delete chunks
        for i in range(n_procs):
            paths.feature(ds, f, i).unlink()
        # delete the empty dir
        paths.feature(ds, f, 0).parent.rmdir()

    # delete candidate chunks
    for i in range(n_procs):
        paths.candidate_pairs(ds, i).unlink()
    # delete the empty dir
    paths.candidate_pairs(ds, 0).parent.rmdir()


def calc_features(ds, feats, cols):
    """Calculates all of the features for a given dataset using multiprocessing"""
    print(f"Started computing the features for {ds}.")
    split_candidates(ds, n_procs)

    for f in feats:
        print(f"- Computing {f} features for {ds}.")
        params = [(ds, f, cols, i) for i in range(n_procs)]

        with Pool(n_procs) as p:
            p.starmap(calc_feature, params)

    merge_datasets(ds, n_procs, feats)


def main(datasets, feats, cols):
    for ds in datasets:
        calc_features(ds, feats, cols)


if __name__ == "__main__":
    main(datasets, features, text_columns)
