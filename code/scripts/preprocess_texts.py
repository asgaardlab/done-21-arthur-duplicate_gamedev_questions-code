import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from multiprocessing import Pool
from gensim.parsing.preprocessing import preprocess_string

from utils import paths, read, save
from utils.consts import datasets, n_procs


def process_html(t):
    """Processes HTML text to replace or remove tags"""
    t = t.lower()
    t = re.sub(r"\n", " ", t)
    t = re.sub(r"<code>.*?</code>", " codesnippet ", t)  # replace code
    t = re.sub(r"<a.*?https?:\/\/.*?[\b\s]?>", " url ", t)  # replace urls
    t = re.sub(r"https?:\/\/.*?(?:[\b\s]|$)", " url ", t)  # replace urls
    t = re.sub(r"<img.*?>", " img ", t)  # replace images
    t = BeautifulSoup(t, features="lxml").get_text()  # remove html tags
    return t


def preprocess_texts(df):
    """Applies the process_html function to different question parts
    and merges them
    """
    df["body"] = df["body"].apply(process_html)
    df["title"] = df["title"].apply(process_html)
    df["answer"] = df["answer"].apply(process_html)
    df["title_body"] = df.title + " " + df.body
    df["title_body_tags"] = df.title_body + " " + df.tags
    df["title_body_tags_answer"] = df.title_body_tags + " " + df.answer
    return df


def tokenize_texts(df):
    """Tokenizes each question part using preprocess_string and merges them"""
    df["body"] = df["body"].apply(preprocess_string)
    df["title"] = df["title"].apply(preprocess_string)
    df["tags"] = df["tags"].apply(preprocess_string)
    df["answer"] = df["answer"].apply(preprocess_string)
    df["title_body"] = df.title + df.body
    df["title_body_tags"] = df.title_body + df.tags
    df["title_body_tags_answer"] = df.title_body_tags + df.answer
    return df


def map_pool(df, f):
    """Splits the dataframe into chunks and
    maps the function using multiprocessing
    """
    dfs = np.array_split(df, n_procs)

    with Pool(n_procs) as p:
        dfs = p.map(f, dfs)

    df = pd.concat(dfs)
    return df


def get_questions(ds):
    """Reads the set of questions for a dataset and normalizes it"""
    df = read(paths.question_texts(ds))

    assert not df.isna().any().any(), "NaN columns in data!"

    df.dropna(inplace=True)
    df = df.drop_duplicates("id")

    # get the ordered index of each question in the dataframe
    df = df.reset_index(drop=True).reset_index()
    df = df.rename(columns={"index": "corpus_index"})

    return df


def select_best_answers(answers):
    """Selects the best answer for a given question
    based on the heuristic:
    accepted answer > highest score > posted first
    """
    answers["max_score"] = answers.groupby("question_id").score.apply(
        lambda s: s == max(s)
    )
    answers["posted_first"] = answers.groupby("question_id").post_date.apply(
        lambda s: s == min(s)
    )

    # subset of answers that *can* be the best one
    best_answers = answers[answers.accepted | answers.max_score | answers.posted_first]

    # sort the answers according to the criteria
    sort_cols = ["accepted", "max_score", "posted_first"]
    best_answers = best_answers.sort_values(sort_cols, ascending=False)

    # the first sorted answer is the "best" one
    best_answers = best_answers.groupby("question_id").first().reset_index()

    assert (
        best_answers.groupby("question_id").apply(len) == 1
    ).all(), "Some questions have more than one answer!"
    assert answers.question_id.isin(best_answers.question_id).all(), "Missing answers!"
    assert (
        answers[answers.accepted].question_id.isin(best_answers.question_id).all()
    ), "Missing accepted answers!"

    return best_answers


def get_answers(ds):
    """Reads the set of answers for a dataset, selects the best one
    and normalizes them
    """
    df = read(paths.answer_texts(ds))
    df = df.drop_duplicates()
    df = select_best_answers(df)
    df = df[["question_id", "body"]]
    df.columns = ["id", "answer"]
    df = df.reset_index(drop=True)
    return df


def questions_with_answers(ds):
    """Reads questions and answers for a given dataset and merges them"""
    questions = get_questions(ds)
    answers = get_answers(ds)

    questions = questions.merge(answers, on="id", how="left")
    questions["answer"] = questions.answer.fillna("")

    assert (questions.groupby("id").apply(len) == 1).all(), "Duplicate questions!"

    questions = questions.set_index("id")

    return questions


def preprocess_questions(ds):
    """Reads questions with answers for a dataset and preprocesses their texts"""
    print(f"- Preprocessing texts for {ds}")
    df = questions_with_answers(ds)

    df = map_pool(df, preprocess_texts)
    save(df, paths.corpus(ds, tokenized=False))

    df = map_pool(df, tokenize_texts)
    save(df, paths.corpus(ds, tokenized=True))


def main(datasets):
    print(f"Preprocessing texts.")
    for ds in datasets:
        preprocess_questions(ds)
    print(f"Finished preprocessing texts.")


if __name__ == "__main__":
    main(datasets)
