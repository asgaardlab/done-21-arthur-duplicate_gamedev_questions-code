import pandas as pd

from utils import paths, save, read
from utils.consts import gamedev_tags, so_sample_seeds


def sample_stackoverflow(sample_num, seed):
    """Creates a sample from the StackOverflow dataset of similar size to the game dev. datasets"""

    def select_dup_pairs():
        """Selects a number of dup pairs equal to the mean of dup pairs in game dev datasets"""
        print("-- Selecting dup pairs")
        len_pairs_se = len(read(paths.dup_pairs("gamedev_se")))
        len_pairs_so = len(read(paths.dup_pairs("gamedev_so")))
        len_pairs = (len_pairs_se + len_pairs_so) // 2

        pairs = read(paths.dup_pairs("stackoverflow"))
        pairs = pairs.drop_duplicates()
        pairs = pairs.sample(n=len_pairs, random_state=seed).reset_index(drop=True)
        save(pairs, paths.dup_pairs(f"so_samples/sample_{sample_num}"))

    def select_questions():
        """Selects a number of questions equal to the mean questions in game dev datasets"""
        print("-- Selecting questions")
        pairs = read(paths.dup_pairs(f"so_samples/sample_{sample_num}"))

        len_se = len(read(paths.question_texts("gamedev_se")))
        len_so = len(read(paths.question_texts("gamedev_so")))
        sample_size = (len_se + len_so) // 2
        remaining_questions = sample_size

        dfs = []

        n_splits = len(list(paths.corpus_dir("stackoverflow").glob("question_texts*")))

        for i in range(n_splits):
            print(i, end="\r")
            # samples to select in this split
            n_samples = (sample_size // n_splits) + int(i < sample_size % n_splits)

            # if there is only one split we don't have suffixes
            if n_splits == 1:
                i = None

            df = read(paths.question_texts("stackoverflow", i))

            # questions in the pre-selected dup pairs
            is_in_pairs = df.id.isin(pairs.main_id) | df.id.isin(pairs.dup_id)

            df_dups = df[is_in_pairs]
            # questions not in pairs
            df = df[~is_in_pairs]

            # random questions to sample from this split
            to_sample = n_samples - len(df_dups)
            sample = df.sample(n=to_sample, random_state=seed)

            df = pd.concat([sample, df_dups]).reset_index(drop=True)
            dfs.append(df)
            print(" " * 50, end="\r")

        df = pd.concat(dfs)
        df = df.drop_duplicates("id")
        df = df.reset_index(drop=True)

        save(df, paths.question_texts(f"so_samples/sample_{sample_num}"))

    def select_answers():
        """Selects only the answers that have questions in the sampled dataset"""
        print("-- Selecting answers")
        qids = read(paths.question_texts(f"so_samples/sample_{sample_num}")).id

        df = []
        n_splits = len(list(paths.corpus_dir("stackoverflow").glob("answer_texts*")))

        for i in range(n_splits):
            print(i, end="\r")

            # if there is only one split we don't have suffixes
            if n_splits == 1:
                i = None

            split = read(paths.answer_texts("stackoverflow", i))

            df_split = split[split.question_id.isin(qids)]
            df.append(df_split)
            print(" " * 50, end="\r")

        df = pd.concat(df)
        df = df.reset_index(drop=True)
        save(df, paths.answer_texts(f"so_samples/sample_{sample_num}"))

    select_dup_pairs()
    select_questions()
    select_answers()


def select_gamedev(tags):
    """Selects posts related to game dev based on the given tags"""

    def select_questions():
        """Selects game dev questions from each split if they contain one of the tags"""
        print("-- Selecting questions")
        df = []
        n_splits = len(list(paths.corpus_dir("stackoverflow").glob("question_texts*")))

        for i in range(n_splits):
            # if there is only one split we don't have suffixes
            if n_splits == 1:
                i = None

            split = read(paths.question_texts("stackoverflow", i))

            for t in tags:
                print(i, t, end="\r")
                # selects questions that have tag 't' in the list
                tag_in_list = lambda ts: t.lower() in ts.lower().split(",")
                df_tag = split[split.tags.apply(tag_in_list)]
                df.append(df_tag)
                print(" " * 50, end="\r")

        df = pd.concat(df)
        df = df.drop_duplicates("id").reset_index(drop=True)
        save(df, paths.question_texts("gamedev_so"))

    def select_answers():
        """Selects only the answers that have questions in the game dev dataset"""
        print("-- Selecting answers")
        qids = read(paths.question_texts("gamedev_so")).id

        df = []
        n_splits = len(list(paths.corpus_dir("stackoverflow").glob("answer_texts*")))

        for i in range(n_splits):
            print(i, end="\r")

            # if there is only one split we don't have suffixes
            if n_splits == 1:
                i = None

            split = read(paths.answer_texts("stackoverflow", i))

            df_split = split[split.question_id.isin(qids)]
            df.append(df_split)

        df = pd.concat(df)
        df = df.reset_index(drop=True)
        save(df, paths.answer_texts("gamedev_so"))

    def select_dup_pairs():
        """Selects only the dup pairs that have both questions in the game dev dataset"""
        print("-- Selecting dup pairs")
        qids = read(paths.question_texts("gamedev_so")).id

        pairs = read(paths.dup_pairs("stackoverflow"))
        pairs = pairs[pairs.main_id.isin(qids) & pairs.dup_id.isin(qids)]
        pairs = pairs.drop_duplicates()
        pairs = pairs.reset_index(drop=True)

        save(pairs, paths.dup_pairs("gamedev_so"))

    select_questions()
    select_answers()
    select_dup_pairs()


def select_so_samples(seeds):
    """Select one sample for each given seed"""
    for i, seed in enumerate(seeds):
        print(f"- Selecting sample {i}")
        sample_stackoverflow(i, seed)


def main(tags_gamedev, seeds):
    print("Selecting Game Dev. SO questions")
    select_gamedev(tags_gamedev)
    print("Sampling Stack Overflow")
    select_so_samples(seeds)


if __name__ == "__main__":
    main(gamedev_tags, so_sample_seeds)
