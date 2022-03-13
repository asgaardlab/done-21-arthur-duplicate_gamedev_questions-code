from termcolor import colored

from utils.consts import *
from extract_data import extract_xml_datasets
from select_samples import select_gamedev, select_so_samples
from preprocess_texts import preprocess_questions
from extract_ids import extract_all_ids
from make_embeddings import train_all_models
from evaluate_similarities import (
    dup_pair_ranks,
    calculate_recall_rates
)
from candidate_selection import select_candidates_multi
from create_features import calc_features
from merge_features import make_sets_multi
from hp_tuning import tune_multiple_sets
from train_classifiers import (
    train_classifiers,
    misclassified_dups,
    candidate_performance,
    cross_dataset_performance,
)

if __name__ == "__main__":
    print(colored("Extracting XML data", "red"))
    extract_xml_datasets(["gamedev_se", "stackoverflow"])
    print()
    print(colored("Selecting Game Dev. SO questions", "red"))
    select_gamedev(gamedev_tags)
    print()
    print(colored("Sampling Stack Overflow", "red"))
    select_so_samples(so_sample_seeds)
    print()
    print(colored(f"Preprocessing texts.", "red"))
    for ds in datasets:
        preprocess_questions(ds)
    print()
    print(colored("Extracting IDs.", "red"))
    for ds in datasets:
        extract_all_ids(ds, noise_percentage, split_percentage)
    print()
    print(colored("Training feature models", "red"))
    for ds in datasets:
        train_all_models(ds, features, text_columns)
    print()
    print(colored("Computing pair ranks", "red"))
    for ds in datasets:
        dup_pair_ranks(ds, features, text_columns)
        calculate_recall_rates(ds, features, text_columns)
    print()
    print(colored("Selecting candidates", "red"))
    for ds in datasets:
        select_candidates_multi(ds, n_candidates, undersampling_percentages)
    print()
    print(colored("Calculating features", "red"))
    for ds in datasets:
        calc_features(ds, features, text_columns)
    print()
    print(colored("Creating train and test sets", "red"))
    make_sets_multi(datasets, n_candidates, undersampling_percentages, features)
    print()
    print(colored("Tuning Hyperparameters", "red"))
    tune_multiple_sets(datasets, n_candidates, undersampling_percentages)
    print()
    print(colored("Training classifiers", "red"))
    for ds in datasets:
        train_classifiers(ds, n_candidates, undersampling_percentages)
    print()
    print(colored("Selecting misclassified duplicates", "red"))
    for ds in ['gamedev_se', 'gamedev_so', 'so_samples/sample_0']:
        misclassified_dups(ds, best_candidates, best_undersampling)
    print()
    print(colored("Evaluating classifiers", "red"))
    candidate_performance(datasets, n_candidates, best_undersampling)
    cross_dataset_performance(
        datasets, best_candidates, best_undersampling
    )
