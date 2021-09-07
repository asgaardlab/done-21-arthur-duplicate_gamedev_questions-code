Each dataset we used in our study has its own directory and follows the same structure. For the data sampled from Stack Overflow (`so_samples/`), each sample has its own directory inside the main directory.

We stored all of the data files in parquet format to allow for faster reading/writing and reduced storage. More information about parquet can be found [here](https://parquet.apache.org/documentation/latest/). There are several libraries capable of reading parquet files. In our code, we use the `arrow` library in [Python](https://arrow.readthedocs.io/en/latest/) (along with [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html)) and [R](https://arrow.apache.org/docs/r/) to read the files. 

Each dataset directory contains the following structure:

- `raw/` - Contains the raw files downloaded directly from the [Stack Exchange data dump](https://archive.org/details/stackexchange).
- `corpus/`
	- `question_texts.parquet` - Contains all of the text and data related to the questions collected for the dataset.
	- `answer_texts.parquet` -  Contains all of the text and data related to the answers collected for the dataset.
	- `corpus.parquet` - Contains the the texts parts for comparing questions with HTML tags removed but prior to tokenization (in a string format). 
	- `corpus_tokenized.parquet` - Contains the the texts parts for comparing questions in tokenized format (as a list of tokens as opposed to a string).
- `question_ids/`
	- `all_question_ids.parquet` - Contains the IDs of all of the questions in the dataset.
	- `answered_question_ids.parquet` - Contains the IDs of all of the answered questions in the dataset.
	- `duplicate_question_ids.paruqet` - Contains the IDs of all of the duplicate questions in the dataset (questions that are marked as duplicates of others).
	- `main_question_ids.parquet` - Contains the IDs of the main questions in the dataset (questions that form a duplicate pair with duplicate questions).
	- `dup_pairs.parquet` - Contains the IDs of the pairs of duplicate questions (duplicate + main question) in the dataset.
	- `comparison_question_ids.parquet` - Contains the IDs of the questions that are eligible to be used when comparing with duplicate questions (i.e., have at least one answer or are marked as a main question).
	- `noise_question_ids.parquet` - Contains the IDs of questions randomly sampled as "noise" to use in the train set.
	- `test_dup_ids.parquet` - Contains the IDs of the duplicate questions belonging to the test set.
	- `train_dup_ids.parquet` - Contains the IDs of the duplicate questions belonging to the train set.
	- `candidate_pairs/` - Contains all of the files with the pre-defined candidate pairs to be used in the train and test sets. Train candidates are split depending on the number of candidates chosen for each set and the percentage of true duplicates in them.
- `embeddings/` - Contains the embeddings we extracted for each question in the dataset using different techniques. The embeddings for each technique are stored in separate subdirectories containing one file for each question part (e.g., title/body/tags).
- `features/` - Contains the files with the values of features computed for each candidate pair according to the techniques we used for comparing the questions. Each technique has its own file.
- `test_sets/` - Contains the test sets we used for evaluating classifier models. The test sets are the junction of test candidates with their corresponding features.
- `train_sets/` - Contains the train sets we used for training classifier models. The train sets are the junction of train candidates sets with their corresponding features.
- `cv_results/` - Contains the random search cross-validation results for each classifier trained on different train sets.

All of the files used for analyzing the results are in the analysis directory (`analysis/`). The directory contains the following files:

- `candidates_evaluation.parquet` - Contains the results we obtained for classifiers trained and evaluated on different numbers of candidates.
- `cross_dataset_performance.parquet` - Contains the results we obtained when evaluating classifiers on datasets different than the ones used for training them.
- `pair_ranks_summary.parquet` - Contains a summary of the recall-rate@k measures for different techniques for all duplicate pairs
- `duplicate_pair_ranks/` - Contains one directory for each dataset, each containing the parquet files with the results we obtained for ranking duplicate pairs with each technique.
- `misclassified_duplicates` - Contains datasets with the duplicate pairs that were misclassified by the classifiers, along with the analysis for the reason for the wrong classification.
