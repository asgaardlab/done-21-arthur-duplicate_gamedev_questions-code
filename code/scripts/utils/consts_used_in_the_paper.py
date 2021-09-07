import multiprocessing

# tags used for selecting gamedev questions on Stack Overflow
gamedev_tags = [
    "game-engine",
    "game-physics",
    "game-development",
    "gameobject",
    "2d-games",
    "unreal-engine4",
    "unreal-blueprint",
    "unreal-development-kit",
    "unrealscript",
    "unity3d",
    "unity5",
    "unity5.3",
    "unity3d-mecanim",
    "unity3d-terrain",
    "unityscript",
    "unity3d-2dtools",
    "unity3d-unet",
    "unity-webgl",
    "unity2d",
    "unity-editor",
    "unity3d-editor",
    "unity-networking",
    "unity3d-gui",
    "unity-ui",
    "unity3d-5",
]

# Seeds used for sampling Stack Overflow questions
so_sample_seeds = [5129, 1011, 3692, 2420, 5815]

# names of the Stack Overflow samples
so_samples = [f"so_samples/sample_{i}" for i, _ in enumerate(so_sample_seeds)]

gamedev_datasets = [
    "gamedev_se",
    "gamedev_so",
]

datasets = gamedev_datasets + so_samples

# Similarity measures / features used in the classifiers
features = [
    "jaccard",
    "tfidf",
    "bm25",
    "topic",
    "doc2vec",
    'bertoverflow',
    'mpnet'       
]

text_columns = [
    "title",
    "body",
    "tags",
    "title_body",
    "title_body_tags",
    "title_body_tags_answer",
]

# number of CPU cores to use
n_procs = 7

# percentage of fake duplicate questions in the train set
noise_percentage = 0.2

# candidates used for training and evaluating classifiers
n_candidates = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000]

# percentages used for undersampling the train sed
undersampling_percentages = [0.01]
# percentage of the train/test split
split_percentage = 0.2

# best values we found for the undersampling / number of selected candidates
best_undersampling = 0.01
best_candidates = 1500

# number of iterations used during random HP tuning
search_n_iters = 30
