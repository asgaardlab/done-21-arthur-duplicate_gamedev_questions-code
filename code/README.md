The code is split between [scripts](scripts/) and [notebooks](notebooks/). The scripts directory contains the main code used for the project, while the notebooks are mainly used for data analysis and visualization.

We describe each of the files for each directory below.

### Scripts

The main directory contains scripts that can be executed directly from the terminal and that are part of the main pipeline for the project.

We list each of the scripts below in the order that they should be executed in order to run the full pipeline and briefly explain what each of them does. To execute all of them in sequence, use the script [`full_pipeline`](scripts/full_pipeline.py).

1. [`extract_data.py`](scripts/extract_data.py) - Extracts and processes the data from the XML files for each website. Reads XML files from `data/{dataset}/raw` directories and saves `*_texts.parquet` datasets to `data/{dataset}/corpus` directories.

2. [`select_samples.py`](scripts/select_samples.py) - Selects samples of posts from Stack Overflow using pre-defined question tags. Reads datasets from `data/stackoverflow/corpus` and saves the corresponding sampled datasets to `data/{dataset}/corpus` directories.

3. [`preprocess_texts.py`](scripts/preprocess_texts.py) - Applies preprocessing functions to posts texts for each dataset, while also merging data from answers to questions. Reads `data/{dataset}/corpus/*_texts.parquet` files and saves `corpus*.parquet` datasets to the same directory.

4. [`extract_ids.py`](scripts/extract_ids.py) - Extract the IDs of some question groups for easier referencing in future steps. Reads `data/{dataset}/corpus.parquet` files and saves IDs to `data/{dataset}/question_ids` directories.

5. [`make_embeddings.py`](scripts/make_embeddings.py) - Trains models for the similarity measures we used for comparing questions and computes embeddings based on them. Reads `data/{dataset}/corpus/corpus*.parquet` files, saves models to `models/{dataset}/*` directories and embeddings to `data/{dataset}/embeddings` directories.

6. [`evaluate_similarities.py`](scripts/evaluate_similarities.py) - Evaluates duplicate question pairs using each of the similarity measures (RQ1 in our paper). Reads 
question IDs from `data/{dataset}/question_ids` directories and saves the results to the `data/analysis/duplicate_pair_ranks` directory.

7. [`candidate_selection.py`](scripts/candidate_selection.py) - Selects candidates for training and evaluating binary classifiers. Reads question IDs from `data/{dataset}/question_ids` directories and save the IDs of candidate pairs to `data/{dataset}/question_ids/candidate_pais` directories.

8. [`create_features.py`](scripts/create_features.py) - Creates the similarity features for each similarity measure for each candidate pair. Reads candidate pair IDs from `data/{dataset}/question_ids/candidate_pairs` and embeddings from `data/{dataset}/embeddings/*` and saves the resulting features in `data/{dataset}/features/*`.

9. [`merge_features.py`](scripts/merge_features.py) - Creates the datasets for training models by merging all of the features to the candidate pair IDs. Reads candidate pair IDs from `data/{dataset}/question_ids/candidate_pairs` and features from `data/{dataset}/features/*` and saves the resulting train and test sets in `data/{dataset}/train_sets` and `data/{dataset}/test_sets` directories.

10. [`hp_tuning.py`](scripts/hp_tuning.py) - Tunes hyperparameters for random forest models. Read train sets from `data/{dataset}/train_sets` directories and saves the results of the search in `data/{datasets}/cv_results` directories.

11. [`train_classifiers.py`](scripts/train_classifiers.py) - Trains the final classifiers and evaluates the results on the test sets. Reads train and test sets from `data/{dataset}/train_sets` and `data/{dataset}/test_sets` directories, hyperparameters from `data/{dataset}/cv_results` directories, and saves models to `models/{dataset}/classifiers` directories and evaluation results in the `data/analysis` directory.


#### Utils

Scripts in the [utils](scripts/utils) directory are imported by the other scripts and implement parameters and functionalities that are used by the other scripts.

- [`paths.py`](scripts/utils/paths.py) - Defines the paths of other files and resources in the project.
- [`consts.py`](scripts/utils/consts.py) - Defines the constant values used in the project.
- [`misc.py`](scripts/utils/misc.py) - Defines a couple of auxiliary functions.
- [`question_comp.py`](scripts/utils/question_comp.py) - Implements the logic for comparing questions and calculating similarities.

##### Models

Scripts in the [models](scripts/utils/models) directory define models and functions to aid training and evaluating their results.

- [`bm25.py`](scripts/utils/models/bm25.py) - An altered implementation of the BM25 algorithm based on the one provided by Gensim 3.8. It adds some functionalities and a few performance improvements.
- [`hp_tuning.py`](scripts/utils/models/hp_tuning.py) - Implements functions (such as the one for creating folds) and parameters (e.g., the HP search grid) for tuning random forest hyperparameters.
- [`misc.py`](scripts/utils/models/misc.py) - Implements the function to split the a dataset between features (X) and the target variable (y).
- [`models.py`](scripts/utils/models/models.py) - Defines the models we used in our project using pre-defined hyperparamets.
- [`scoring.py`](scripts/utils/models/scoring.py) - Implements functions for scoring the models using the recall-rate measures.

### Notebooks

The notebooks are used mainly for analysis and data visualization. There we also included one notebook for computing embeddings using GPUs on Google Colab.

- [`create_tables.py`](notebooks/create_tables.py) - Creates and formats the tables we used in our study using the data obtained from the pipeline described above.
- [`paper_results.ipynb`](notebooks/paper_results.ipynb) - Shows the tables used in our paper.
- [`plot_figures.R`](notebooks/plot_figures.R) - Defines functions to plot the figures presented in the paper using GGPlot.
- [`paper_figures.ipynb`](notebooks/paper_figures.ipynb) - Shows the figures we used in our paper.
- [`make_embeddings_colab_gpu.ipynb`](notebooks/make_embeddings_colab_gpu.ipynb) - Notebook used for computing embeddings using GPUs on Google Colab.
