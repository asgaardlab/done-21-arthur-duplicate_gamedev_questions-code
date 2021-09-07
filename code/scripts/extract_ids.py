import pandas as pd

from utils import paths, read, save
from utils.consts import datasets, split_percentage, noise_percentage


def extract_dup_pair_ids(ds):
    """Extracts the IDs of dup pairs in the dataset"""
    pairs = read(paths.dup_pairs(ds))
    corpus = read(paths.corpus(ds)).reset_index()

    print(f"- {len(pairs)} dup pairs")

    dups = corpus[corpus.id.isin(pairs.dup_id)]
    dups = dups[["id", "corpus_index"]]
    dups = dups.reset_index(drop=True)

    print(f"-- {len(dups)} dups")

    save(dups, paths.duplicate_question_ids(ds))

    main_qs = corpus[corpus.id.isin(pairs.main_id)]
    main_qs = main_qs[["id", "corpus_index"]]
    main_qs = main_qs.reset_index(drop=True)

    print(f"-- {len(main_qs)} main questions")

    save(main_qs, paths.main_question_ids(ds))


def extract_question_ids(ds):
    """Extracts question IDs for all questions in the dataset"""
    df = read(paths.corpus(ds))
    df = df.reset_index()

    print(f"- {len(df)} questions")

    answered_ids = df[df.n_answers > 0][["id", "corpus_index"]]
    answered_ids = answered_ids.reset_index(drop=True)
    save(answered_ids, paths.answered_question_ids(ds))

    print(f"- {len(answered_ids)} answered questions")

    df = df[["id", "corpus_index"]]
    save(df, paths.all_question_ids(ds))

    main_ids = read(paths.main_question_ids(ds))

    # Questions that will be compared = answered + main questions
    comp_ids = pd.concat([answered_ids, main_ids])
    comp_ids = comp_ids.reset_index(drop=True)
    comp_ids = comp_ids.drop_duplicates("id")

    save(comp_ids, paths.comparison_question_ids(ds))


def sample_noise_questions(ds, perc):
    """Samples a percentage of questions to serve as noise in supervised learning"""
    df = read(paths.corpus(ds))
    dups = read(paths.duplicate_question_ids(ds))
    mains = read(paths.main_question_ids(ds))

    df = df.reset_index()[["id", "corpus_index"]]
    # exclude true duplicates and their pairs
    df = df[~df.id.isin(dups.id) & ~df.id.isin(mains.id)]

    # sample a percentage of the number of duplicates
    n_dups = len(dups)
    noise_dups = round(n_dups * perc)
    noise = df.sample(noise_dups, random_state=42).reset_index(drop=True)

    save(noise, paths.noise_question_ids(ds))


def split_train_test_dups(ds, perc):
    """Randomly splits the duplicates that will be used for train and test sets"""
    dups = read(paths.duplicate_question_ids(ds))

    test_dups = dups.sample(frac=perc, random_state=42)
    test_dups = test_dups.reset_index(drop=True)

    save(test_dups, paths.test_dup_ids(ds))

    # remaining dups used for training
    train_dups = dups[~dups.id.isin(test_dups.id)]
    train_dups = train_dups.reset_index(drop=True)

    save(train_dups, paths.train_dup_ids(ds))

    print(f"- {len(train_dups)} dups in the train set")
    print(f"- {len(test_dups)} dups in the test set")


def extract_all_ids(ds, noise_p, split_p):
    """Extract all ids for a given dataset using the functions above"""
    print(f"Extracting IDs for {ds}.")
    extract_dup_pair_ids(ds)
    extract_question_ids(ds)
    sample_noise_questions(ds, noise_p)
    split_train_test_dups(ds, split_p)


def main(datasets, noise_p, split_p):
    print("Extracting IDs.")
    for ds in datasets:
        extract_all_ids(ds, noise_p, split_p)


if __name__ == "__main__":
    main(datasets, noise_percentage, split_percentage)
