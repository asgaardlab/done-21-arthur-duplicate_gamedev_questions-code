import os
from pathlib import Path


def root_dir():
    path = os.path.abspath(os.getcwd())
    root = path.rsplit("code", 1)[0]
    return Path(root)


def percent_to_string(p):
    p = p * 100
    if p > 0:
        p = int(p)
    return str(p).replace(".", "_")


####################
## Data dirs      ##
####################


def data_dir():
    return root_dir() / "data"


def dataset_dir(ds):
    return data_dir() / ds


# Raw data


def raw_dir(ds):
    return dataset_dir(ds) / "raw"


def posts_xml(ds, i=None):
    return raw_dir(ds) / "Posts.xml"


def post_links_xml(ds):
    return raw_dir(ds) / "PostLinks.xml"


def questions_xml(ds, i=None):
    if i is None:
        return raw_dir(ds) / "questions.xml"
    else:
        return raw_dir(ds) / f"questions_{i}.xml"


def answers_xml(ds, i=None):
    if i is None:
        return raw_dir(ds) / "answers.xml"
    else:
        return raw_dir(ds) / f"answers_{i}.xml"


# Question IDs


def ids_dir(ds):
    return dataset_dir(ds) / "question_ids"


def all_question_ids(ds):
    return ids_dir(ds) / "all_question_ids.parquet"


def accepted_answer_ids(ds):
    return ids_dir(ds) / "accepted_answers.parquet"


def dup_pairs(ds):
    return ids_dir(ds) / "dup_pairs.parquet"


def duplicate_question_ids(ds):
    return ids_dir(ds) / "duplicate_question_ids.parquet"


def main_question_ids(ds):
    return ids_dir(ds) / "main_question_ids.parquet"


def noise_question_ids(ds):
    return ids_dir(ds) / "noise_question_ids.parquet"


def comparison_question_ids(ds):
    return ids_dir(ds) / "comparison_question_ids.parquet"


def answered_question_ids(ds):
    return ids_dir(ds) / "answered_question_ids.parquet"


def train_dup_ids(ds):
    return ids_dir(ds) / "train_dup_ids.parquet"


def test_dup_ids(ds):
    return ids_dir(ds) / "test_dup_ids.parquet"


# Corpus


def corpus_dir(ds):
    return dataset_dir(ds) / "corpus"


def question_texts(ds, i=None):
    if i is None:
        return corpus_dir(ds) / "question_texts.parquet"
    else:
        return corpus_dir(ds) / f"question_texts_{i}.parquet"


def answer_texts(ds, i=None):
    if i is None:
        return corpus_dir(ds) / "answer_texts.parquet"
    else:
        return corpus_dir(ds) / f"answer_texts_{i}.parquet"


def corpus(ds, tokenized=True):
    if tokenized:
        return corpus_dir(ds) / "corpus_tokenized.parquet"
    else:
        return corpus_dir(ds) / "corpus.parquet"


# Embeddings


def embeddings_dir(ds):
    return dataset_dir(ds) / "embeddings"


def embedding_dir(ds, m):
    return embeddings_dir(ds) / m


def embedding(ds, m, c):
    return embedding_dir(ds, m) / f"{c}.{m}.npz"


# Features


def features_dir(ds):
    return dataset_dir(ds) / "features"


def feature(ds, f, i=None):
    if i is None:
        save = features_dir(ds) / f"{f}.parquet"
    else:
        save = features_dir(ds) / f / f"{i}.parquet"
    return save


# Candidates


def candidate_pairs_dir(ds):
    return ids_dir(ds) / "candidate_pairs"


def candidate_pairs(ds, i=None):
    if i is None:
        path = candidate_pairs_dir(ds) / "candidate_pairs.parquet"
    else:
        path = candidate_pairs_dir(ds) / "split" / f"candidate_pairs_{i}.parquet"
    return path


def train_candidate_pairs(ds, c, p=None, i=None):
    filename = f"train_candidate_pairs_{c}_candidates"

    if p is not None:
        p = percent_to_string(p)
        filename += f"_{p}_perc_dups"
    if i is not None:
        filename += f"_{i}"

    filename += ".parquet"

    return candidate_pairs_dir(ds) / filename


def test_candidate_pairs(ds):
    return candidate_pairs_dir(ds) / "test_candidate_pairs.parquet"


# Train/test sets


def train_sets_dir(ds):
    return dataset_dir(ds) / "train_sets"


def test_sets_dir(ds):
    return dataset_dir(ds) / "test_sets"


def train_set(ds, c, p, i=None):
    p = percent_to_string(p)
    return train_sets_dir(ds) / f"train_{c}_candidates_{p}_perc_dups.parquet"


def test_set(ds):
    return test_sets_dir(ds) / f"test.parquet"


# Cross-val results


def cv_results_dir(ds):
    return dataset_dir(ds) / "cv_results"


def cv_results(ds, c, p):
    p = percent_to_string(p)
    return cv_results_dir(ds) / f"cv_results_{c}_candidates_{p}_perc_dups.parquet"


####################
## Model dirs     ##
####################


def models_dir():
    return root_dir() / "models"


def dataset_models_dir(ds):
    return models_dir() / ds


# Feature models


def feature_model_dir(ds, m):
    return dataset_models_dir(ds) / m


def feature_model(ds, m, c):
    return feature_model_dir(ds, m) / f"{c}.{m}"


# Classifiers


def classifiers_dir(ds):
    return dataset_models_dir(ds) / "classifiers"


def classifier(ds, c, p):
    p = percent_to_string(p)
    return classifiers_dir(ds) / f"classifier_{c}_candidates_{p}_perc_dups.joblib"


####################
## Analysis dirs  ##
####################


def analysis_dir():
    return data_dir() / "analysis"


def analysis_file(f):
    return analysis_dir() / f


def candidates_evaluation():
    return analysis_file("candidates_evaluation.parquet")


def cross_dataset_performance():
    return analysis_file("cross_dataset_performance.parquet")


# Misclassified duplicates


def misclassified_duplicates_dir():
    return analysis_dir() / "misclassified_duplicates"


def misclassified_duplicates(ds):
    return misclassified_duplicates_dir() / f"{ds}_misclassified.parquet"


# Question pair ranks


def pair_ranks_dir():
    return analysis_dir() / "duplicate_pair_ranks"


def pair_ranks_dataset_dir(ds):
    return pair_ranks_dir() / ds


def pair_ranks(ds, m, c):
    return pair_ranks_dataset_dir(ds) / f"{m}_{c}.parquet"


def all_pair_ranks(ds):
    return pair_ranks_dataset_dir(ds) / "all_pair_ranks.parquet"
