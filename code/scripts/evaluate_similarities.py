import numpy as np
import pandas as pd
from multiprocessing import Pool

from utils import paths, read, save, QuestionComp
from utils.consts import datasets, features, text_columns, n_procs


def score_dup_pairs(dups, ds, f, c):
    """Scores a set of duplicates against all other answered/main questions
    in the dataset for a given similarity (feature + question part)
    """

    def merge_scores(df, qs, scores):
        """Adds similarity scores to the dataframe of questions"""
        dup_id, dup_index = df.name
        qs = qs.copy()
        qs["score"] = scores[dup_index]
        qs = qs.drop(columns="corpus_index")
        return qs

    comp_qs = read(paths.comparison_question_ids(ds))[["id", "corpus_index"]]
    qc = QuestionComp(ds, f, c)

    scores = qc.compare(dups["corpus_index"], comp_qs["corpus_index"])

    # replace corpus index with dup index to select correct set of scores
    dups = dups.drop(columns="corpus_index").reset_index(drop=True)
    dups = dups.reset_index()
    dups = dups.rename(columns={"id": "dup_id"})

    merge_dup_scores = lambda df: merge_scores(df, comp_qs, scores)

    scores = dups.groupby(["dup_id", "index"]).apply(merge_dup_scores)
    scores = scores.reset_index()
    scores = scores.drop(columns=["index"])

    return scores


def calculate_scores(ds, f, c):
    """Scores all duplicates against all other answered/main questions
    in the dataset for a given similarity (feature + question part)
    Uses multiprocessing + the score_dup_pairs function
    """
    dups = read(paths.duplicate_question_ids(ds))

    dups = np.array_split(dups, n_procs)
    tups = [(d, ds, f, c) for d in dups]

    with Pool(n_procs) as p:
        dups = p.starmap(score_dup_pairs, tups)

    dups = pd.concat(dups)

    return dups


def rank_dup_pairs(ds, f, c):
    """Ranks dup pairs against all other question pairs
    in the dataset based on the similarity scores for a
    feature + column
    """
    scores = calculate_scores(ds, f, c)

    scores = scores.rename(columns={"id": "main_id"})

    # removes pairs of the same question
    scores = scores[scores.main_id != scores.dup_id]

    pairs = read(paths.dup_pairs(ds))
    pairs["is_dup"] = True

    scores = scores.merge(pairs, on=["dup_id", "main_id"], how="left")
    scores["is_dup"] = scores.is_dup.fillna(False)

    # get rank
    scores["rank"] = scores.groupby("dup_id").score.rank(ascending=False)

    scores = scores[scores.is_dup]
    scores = scores.reset_index(drop=True)
    scores = scores.drop(columns=["is_dup", "level_2"])

    save(scores, paths.pair_ranks(ds, f, c))


def calculate_recall_rates(ds, feats, cols):
    """Calculate recall-rates@k based on the dup pairs ranks for a given dataset"""

    def recall_rate(df, k):
        """Calculates the recall-rate@k for a dataset using the rank column"""
        # rank <= k -> dup pair in top k results
        has_dup_in_k = lambda r: (r <= k).any()
        return df.groupby("dup_id")["rank"].apply(has_dup_in_k).mean()

    recall_rates = []

    for f in feats:
        for c in cols:
            scores = read(paths.pair_ranks(ds, f, c))

            rates = {
                "feature": f,
                "col": c,
            }

            for i in [5, 10, 20]:
                rates[f"recall-rate@{i}"] = recall_rate(scores, i)

            recall_rates.append(rates)

    recall_rates = pd.DataFrame(recall_rates)
    save(recall_rates, paths.all_pair_ranks(ds))


def dup_pair_ranks(ds, feats, columns):
    """Calculates the ranks of dup pais for all similarity scores in the dataset"""
    print(f"Ranking pair ranks for {ds}")
    res = []
    for f in feats:
        print(f"- Computing pair ranks for {f}")
        for c in columns:
            rank_dup_pairs(ds, f, c)


def main(datasets, feats, columns):
    for ds in datasets:
        dup_pair_ranks(ds, feats, columns)
        calculate_recall_rates(ds, feats, columns)


if __name__ == "__main__":
    main(datasets, features, text_columns)
