import numpy as np
import pandas as pd

from utils import paths, read, save, QuestionComp
from utils.consts import datasets, n_candidates, undersampling_percentages


def select_candidates(ds, dups, comp, n):
    """Selects n candidate questions for each dup from the set of provided questions"""

    def dup_cands(df, ids, scores, n):
        """Selects the n candidates with the highest score for the dup in the df"""
        dup_id, _, dup_index = df.name
        ids = ids.copy()
        ids["score"] = scores[dup_index]
        # remove pairs with same questions
        ids = ids[ids.candidate_id != dup_id].copy()
        ids = ids.sort_values("score", ascending=False)[:n]
        return ids

    def get_scores(ds, dups, comp):
        """Gets the scores according to a pre-defined similarity to use
        when selecting candidate questions
        """
        qc = QuestionComp(ds, "tfidf", "title_body_tags_answer")
        return qc.compare(dups["corpus_index"], comp["corpus_index"])

    scores = get_scores(ds, dups, comp)

    dups = dups.rename(columns={"id": "dup_id", "corpus_index": "dup_corpus_index"})

    # get index of the duplicate in the df
    dups = dups.reset_index()

    comp = comp.rename(
        columns={"id": "candidate_id", "corpus_index": "candidate_corpus_index"}
    )

    get_cands = lambda df: dup_cands(df, comp, scores, n)
    candidates = dups.groupby(["dup_id", "dup_corpus_index", "index"]).apply(get_cands)

    # fix index
    candidates = candidates.reset_index(level="index", drop=True)
    candidates = candidates.reset_index()
    candidates = candidates.drop(columns="level_2")

    same_ids = candidates[candidates.dup_id == candidates.candidate_id]
    assert len(same_ids) == 0, "Some dup IDs are the same as their related IDs!"
    assert (
        not candidates[["dup_id", "candidate_id"]].duplicated().any()
    ), "Duplicated candidates!"

    return candidates


def add_dup_labels(cands, pairs):
    """Adds labels indicating if candidate pairs are duplicates or not"""
    pairs["is_dup"] = True
    cands = cands.merge(
        pairs[["dup_id", "main_id", "is_dup"]],
        left_on=["dup_id", "candidate_id"],
        right_on=["dup_id", "main_id"],
        how="left",
    )
    cands = cands.drop(columns="main_id")
    cands["is_dup"] = cands.is_dup.fillna(False)
    return cands


def add_noise_labels(cands, noise):
    """Adds labels indicating if candidate pairs are noise or not"""
    noise["is_noise"] = True
    noise = noise.rename(columns={"id": "dup_id"})
    cands = cands.merge(noise[["dup_id", "is_noise"]], on="dup_id", how="left")
    cands["is_noise"] = cands.is_noise.fillna(False)
    return cands


def undersample(candidates, perc):
    """Undersamples the set of candidate pairs to achieve a desired percentage
    of true duplicate pairs
    """

    def samples_per_dup():
        """Calculates how many false candidates pairs
        we have to sample for each duplicate question
        """
        neg_dups = negatives["dup_id"].unique()

        # the number of negatives for each positive sample
        negs_per_pos = round(1 / perc - 1)

        # total number of negatives and positives
        n_negs = negs_per_pos * len(positives)
        n_dups = len(neg_dups)

        # lower bound of negatives per duplicate question
        lower_n = n_negs // n_dups

        # adds one additional false sample for each dup
        # until there is no remainder left (n_negs % n_dups)
        # guarantees that we will have exactly n_negs
        # and that dups will have similar numbers of pairs
        return {d: lower_n + int(i < n_negs % n_dups) for i, d in enumerate(neg_dups)}

    def select_samples(dup_df):
        """Selects a number of samples for the duplicate question
        in the df according to the values obtained by samples_per_dup
        """
        return dup_df.sample(dup_samples[dup_df.name], random_state=42)

    positives = candidates[candidates.is_dup]
    negatives = candidates[~candidates.is_dup]
    dup_samples = samples_per_dup()

    negatives = negatives.groupby("dup_id").apply(select_samples).reset_index(drop=True)

    return pd.concat([positives, negatives]).reset_index(drop=True)


def decrease_candidates(candidates, n):
    """Reduces the number of candidate pairs to n
    This function is useful for selecting candidates only once for a large N
    and then reducing the size if we need smaller Ns
    """
    limit_cands = lambda df: df.sort_values("score", ascending=False)[:n]
    return candidates.groupby("dup_id").apply(limit_cands).reset_index(drop=True)


def select_train_candidates_split(ds, dups, n, undersample_perc=None):
    """Selects n train candidates pairs for the given dups of the dataset
    in chunks to save memory space
    For datasets with large numbers of candidates it is easy to
    fill up memory space
    """
    # select candidates for each chunk
    splits = 10
    dups = np.array_split(dups, splits)

    for i, ds in enumerate(dups):
        print(i, end="\r")
        ds = ds.reset_index(drop=True)
        select_train_candidates_single(ds, ds, n, undersample_perc, i)

    # merge candidates sampled above
    candidates = []
    for i in range(splits):
        path = paths.train_candidate_pairs(ds, n, undersample_perc, i)
        candidates.append(read(path))
        path.unlink()

    candidates = pd.concat(candidates).reset_index(drop=True)

    save(candidates, paths.train_candidate_pairs(ds, n, undersample_perc))


def select_train_candidates_single(ds, dups, n, undersample_perc=None, i=None):
    """Selects n train candidates pairs for the given dups of the dataset
    the i parameter allows for saving chunks of candidates separately
    """
    test = read(paths.test_dup_ids(ds))
    comp = read(paths.comparison_question_ids(ds))
    # remove test dups from the comparison questions
    # to avoid leakage (comparing train dups with test dups)
    comp = comp[~comp.id.isin(test.id)]

    candidates = select_candidates(ds, dups, comp, n)

    dup_pairs = read(paths.dup_pairs(ds))
    candidates = add_dup_labels(candidates, dup_pairs)

    noise = read(paths.noise_question_ids(ds))
    candidates = add_noise_labels(candidates, noise)

    if undersample_perc is not None:
        candidates = undersample(candidates, undersample_perc)

    save(candidates, paths.train_candidate_pairs(ds, n, undersample_perc, i))


def select_train_candidates(ds, n, undersample_perc=None):
    """Selects n train candidates for all of the train dups and noise questions in the dataset"""
    train = read(paths.train_dup_ids(ds))
    # noise is added to avoid bias
    noise = read(paths.noise_question_ids(ds))

    dups = pd.concat([train, noise]).reset_index(drop=True)

    print("- Selecting train candidates")

    # for datasets with >= 5000 dups, we split them to save memory space
    if len(dups) < 5000:
        select_train_candidates_single(ds, dups, n, undersample_perc)
    else:
        select_train_candidates_split(ds, dups, n, undersample_perc)


def select_train_candidates_multi(ds, n_candidates, percentages):
    """Selects train candidate pairs for a given dataset
    using different values of n and undersampling percentages
    """
    # we only need to sample the max number of candidates
    # then we can use the decrease_candidates function
    # to limit the number of candidates
    max_candidates = max(n_candidates)
    select_train_candidates(ds, max_candidates)

    candidates = read(paths.train_candidate_pairs(ds, max_candidates))

    print("-- Making train sets")
    for c in n_candidates:
        reduced = decrease_candidates(candidates, c)
        for p in percentages:
            print(f"--- {c}, {p}")
            # undersample the dataset only if the percentage of
            # duplicates is smaller than the undersampling percentage
            if reduced.is_dup.mean() < p:
                sampled = undersample(reduced, p)
            else:
                sampled = reduced

            save(sampled, paths.train_candidate_pairs(ds, c, p))


def select_test_candidates(ds, n):
    """Selects n test candidates for all of the test dups in the dataset"""
    dups = read(paths.test_dup_ids(ds))
    comp = read(paths.comparison_question_ids(ds))

    dup_pairs = read(paths.dup_pairs(ds))

    print("- Selecting test candidates")

    candidates = select_candidates(ds, dups, comp, n)
    candidates = add_dup_labels(candidates, dup_pairs)
    # no noise in the test
    candidates["is_noise"] = False

    save(candidates, paths.test_candidate_pairs(ds))


def select_test_candidates_multi(ds, n_candidates):
    """Selects test candidate pairs for a given dataset
    using different values of n
    """
    # we only need to sample the max number of candidates
    # we can later limit the number of candidates used
    # during the evaluation of the classifiers
    max_candidates = max(n_candidates)
    select_test_candidates(ds, max_candidates)


def merge_candidates(ds, candidates, percentages):
    """Merges all of the sets of candidate pairs into a single dataset
    This way we avoid having to compute and compare features multiple
    times for candidate pairs that appear in many sets of candidates
    """
    print("- Merging all sets of candidates")
    candidate_sets = [read(paths.test_candidate_pairs(ds))]
    for p in percentages:
        for c in candidates:
            candidate_sets.append(read(paths.train_candidate_pairs(ds, c, p)))

    candidate_sets = pd.concat(candidate_sets)
    candidate_sets = candidate_sets.drop_duplicates()
    candidate_sets = candidate_sets.reset_index(drop=True)

    save(candidate_sets, paths.candidate_pairs(ds))


def select_candidates_multi(ds, ns, ps):
    """Selects candidate sets for multiple ns and undersampling percentages"""
    print(f"Selecting candidates for {ds}.")
    select_train_candidates_multi(ds, ns, ps)
    select_test_candidates_multi(ds, ns)
    merge_candidates(ds, ns, ps)


def select_candidates_single(ds, n, p):
    """Selects candidate sets for a single value of n and undersampling percentage"""
    print(f"Selecting candidates for {ds}.")
    select_train_candidates(ds, n, p)
    select_test_candidates(ds, n)
    merge_candidates(ds, [n], [p])


def main(datasets, ns, ps):
    # we select multiple candidate sets for the datasets
    # for our analysis
    for ds in datasets:
        select_candidates_multi(ds, ns, ps)


if __name__ == "__main__":
    main(datasets, n_candidates, undersampling_percentages)
