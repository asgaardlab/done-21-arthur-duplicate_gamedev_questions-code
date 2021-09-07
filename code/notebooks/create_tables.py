# Importing scripts from the other dir
import os
import sys

path = os.path.abspath(os.getcwd())
scripts = path.rsplit('code', 1)[0] + 'code/scripts'
sys.path.insert(0, scripts)

import numpy as np
import pandas as pd
from utils import read, paths
from utils.consts import datasets, gamedev_datasets

def dataset_name(ds):
    """Returns the pretty version of the dataset name based
    on the string that represents it
    """
    if ds == 'gamedev_se':
        name = 'Game Dev. Stack Exchange'
    elif ds == 'gamedev_so':
        name = 'Stack Overflow (Game dev.)'
    else:
        name = 'Stack Overflow (General dev.)'
    return name

def dup_pair_similarity_summary():
    """Creates a summary table with all the recall-rates for all similarity scores
    and formats it
    """
    summary = []
    for ds in datasets:
        df = read(paths.all_pair_ranks(ds))
    
        sim_names = {
            "jaccard": "Jaccard",
            "bm25": "BM25",
            "tfidf": "TF-IDF",
            "doc2vec": "Doc2Vec",
            "topic": "Topic",
            "bertoverflow": "BERTOverflow",
            "mpnet": "MPNet",
        }
    
        # order in which to show results
        score_order = ["recall-rate@5", "recall-rate@10", "recall-rate@20"]
        doc_order = [
            "title",
            "body",
            "tags",
            "title_body",
            "title_body_tags",
            "title_body_tags_answer",
        ]
        sim_order = [
            "Jaccard",
            "TF-IDF",
            "BM25",
            "Topic",
            "Doc2Vec",
            "BERTOverflow",
            "MPNet",
        ]
    
        # rename features
        df.feature = df.feature.apply(lambda f: sim_names[f])
    
        df = df.pivot_table(values=score_order, columns="feature", index="col")
        df = df.transpose()
        df.columns = list(df.columns)
    
        df.index = df.index.set_names(["Score", "Similarity"])
        df = df[doc_order]
        # change column names to fit page
        df.columns = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)"]
        df = df.reindex(sim_order, level=1)
        df = df.reindex(score_order, level=0)
        # turn recall-rates into percentages with 2 decimals
        df = (df * 100).round(2)
        
        df["Dataset"] = dataset_name(ds)
    
        summary.append(df)
    
    summary = pd.concat(summary)
    summary = summary.reset_index()
    
    # calculate std and mean
    summary = summary.groupby(['Dataset', 'Score', 'Similarity']).agg([np.mean, np.std])
    
    # format values
    for c in set(summary.columns.get_level_values(0)):
        summary[(c, 'mean')] = summary[(c, 'mean')].apply(lambda f: "{:.2f}".format(f))
        summary[(c, 'std')] = summary[(c, 'std')].apply(lambda f: "" if pd.isna(f) else "({:.2f})".format(round(f, 2)))
        summary[(c, 'mean')] = summary[(c, 'mean')] + " " + summary[(c, 'std')]
        
    # drop columns + fix order
    summary = summary.iloc[:,summary.columns.get_level_values(1) == 'mean']
    summary.columns = summary.columns.droplevel(1)
    summary = summary.reindex(sim_order, level=2)
    summary = summary.reindex(score_order, level=1)
    
    return summary

def candidate_evaluation_summary():
    """Formats the dataframe containing the evaluation results for different numbers of candidates"""
    df = read(paths.candidates_evaluation())
    df['dataset'] = df['dataset_train'].apply(dataset_name)
    
    df = df.drop(columns='dataset_train')
    
    for c in [c for c in df.columns if 'rr' in c]:
        df[c] = (df[c]*100).round(2)
        
    df = df.groupby(['dataset', 'candidates']).agg([np.mean, np.std])
    
    # format values
    for c in set(df.columns.get_level_values(0)):
        df[(c, 'mean')] = df[(c, 'mean')].apply(lambda f: "{:.2f}".format(f))
        df[(c, 'std')] = df[(c, 'std')].apply(lambda f: "" if pd.isna(f) else "({:.2f})".format(round(f, 2)))
        df[(c, 'mean')] = df[(c, 'mean')] + " " + df[(c, 'std')]
        
    # drop columns + fix order and naming
    df = df.iloc[:,df.columns.get_level_values(1) == 'mean']
    df.columns = df.columns.droplevel(1)
    df.columns = pd.MultiIndex.from_product([['Recall-rate@'], [c[3:] for c in df.columns]])
    return df

def dataset_sizes():
    """Summarizes the sizes of the datasets used in the study"""
    df = []
    for ds in datasets:
        n_qs = len(read(paths.all_question_ids(ds)))
        pairs = read(paths.dup_pairs(ds))
        n_dups = len(read(paths.duplicate_question_ids(ds)))
        
        if ds == 'gamedev_se':
            source = 'Stack Exchange'
        else:
            source = 'Stack Overflow'
            
        if 'so_samples' not in ds:
            topic = 'Game development'
        else:
            topic = 'General development'
        
        df.append({
            'Source': source,
            'Topic': topic,
            'Questions': n_qs,
            'Non-duplicates': n_qs-n_dups,
            'Duplicates': n_dups,
            'Pairs': len(pairs)
        })
        
    df = pd.DataFrame(df)
    df = df.groupby(['Source', 'Topic']).agg([np.mean, np.std])
    df = df.fillna(0)
    
    if (df.iloc[:,df.columns.get_level_values(1) == 'std'] == 0).all().all():
        print('All datasets have std == 0')
    
    df = df.iloc[:,df.columns.get_level_values(1) == 'mean']
    df.columns = df.columns.droplevel(1)
    df['dup_perc'] = df['Duplicates']/df['Questions']*100
    df['dup_perc'] = df['dup_perc'].apply(lambda x: round(x, 1))
    
    for c in df.columns:
        if c != 'dup_perc':
            df[c] = df[c].apply(lambda i: "{:,}".format(i))
            
    df['Duplicates'] = df['Duplicates'] + ' (' + df['dup_perc'].apply(str) + '%)'
    df = df.drop(columns='dup_perc')
    
    return df


def cross_dataset_summary():
    """Creates a table summarizing the performance of the classifiers
    when testing on other datasets
    """
    df = read(paths.cross_dataset_performance())
    df['dataset_train'] = df['dataset_train'].apply(dataset_name)
    df['dataset_test'] = df['dataset_test'].apply(dataset_name)

    for c in [c for c in df.columns if 'rr' in c]:
        df[c] = (df[c]*100).round(2)

    df = df.groupby(['dataset_train', 'dataset_test']).agg([np.mean, np.std])

    # format values
    for c in set(df.columns.get_level_values(0)):
        df[(c, 'mean')] = df[(c, 'mean')].apply(lambda f: "{:.2f}".format(f))
        df[(c, 'std')] = df[(c, 'std')].apply(lambda f: "" if pd.isna(f) else "({:.2f})".format(round(f, 2)))
        df[(c, 'mean')] = df[(c, 'mean')] + " " + df[(c, 'std')]

    # drop columns + fix order and naming
    df = df.iloc[:,df.columns.get_level_values(1) == 'mean']
    df.columns = df.columns.droplevel(1)
    df = df.sort_index(level=1)
    df.columns = pd.MultiIndex.from_product([['Recall-rate@'], [c[3:] for c in df.columns]])
    
    return df

def misclassified_summary():
    """Creates a table summarizing the duplicates that were misclassified by our classifiers"""
    def to_percentage(a, b):
        p = round(a/b*100)
        return f'({p}%)'
    
    df = {}
    
    df['Description'] = [
        'Duplicate pairs in test set',
        'Misclassified duplicate pairs',
        'Main question not in the list of candidates',
        'Top ranked question is an unlabelled duplicate',
        'Main question discusses a more general topic'
    ]
    
    for ds in gamedev_datasets + ['so_samples/sample_0']:
        test = read(paths.test_set(ds))
        dups_in_test = len(test.dup_id.unique())
        
        if 'so_samples' in ds:
            missed = read(paths.misclassified_duplicates('so_sample'))
        else:
            missed = read(paths.misclassified_duplicates(ds))
        
        n_misclassified = len(missed)
        no_candidate = len(missed[~missed.has_dup])
        top_ranked_dup = len(missed[missed.top_ranked_is_dup])
        main_more_general = len(missed[missed.main_more_generic])
        
        misc_perc = to_percentage(n_misclassified, n_misclassified)
        no_candidate_perc = to_percentage(no_candidate, n_misclassified)
        top_ranked_perc = to_percentage(top_ranked_dup, n_misclassified)
        more_general_perc = to_percentage(main_more_general, n_misclassified)
    
        n_misclassified = str(n_misclassified) + ' ' + misc_perc
        no_candidate = str(no_candidate) + ' ' + no_candidate_perc
        top_ranked_dup = str(top_ranked_dup) + ' ' + top_ranked_perc
        main_more_general = str(main_more_general) + ' ' + more_general_perc
        
        ds = dataset_name(ds)
        
        df[ds] = [dups_in_test, n_misclassified, no_candidate, top_ranked_dup, main_more_general]
    
    df = pd.DataFrame(df)
    return df