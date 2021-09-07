import joblib
from scipy.sparse import save_npz, csr_matrix
from gensim.matutils import sparse2full
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import paths, read, save, make_dir
from utils.consts import datasets, text_columns, features
from utils.models import (
    doc2vec_model,
    bm25_model,
    lda_model,
    bertoverflow_model,
    mpnet_model,
)


def train_bm25(ds, cols):
    """Trains and saves a BM25 model for each text column in the dataset"""
    print("- Training BM25 models")
    feature_name = "bm25"

    corpus = read(paths.corpus(ds))

    make_dir(paths.feature_model_dir(ds, feature_name))

    for c in cols:
        model_save_path = paths.feature_model(ds, feature_name, c)
        bm25 = bm25_model(corpus[c])
        bm25.save(model_save_path)


def train_doc2vec(ds, cols):
    """Trains and saves a Doc2Vec model for each text column in the dataset
    Also saves document embeddings learned by the models.
    """

    def train_doc2vec_from_file(c):
        """Trains a Doc2Vec model from a file containing the corpus (increased performance)"""
        training_corpus = paths.feature_model_dir(ds, feature_name) / (c + ".txt")

        # create a space-separated text file
        with open(training_corpus, "w") as f:
            for l in corpus[c].apply(lambda s: " ".join(s)):
                f.write(l + "\n")

        model = doc2vec_model(str(training_corpus))
        training_corpus.unlink()
        return model

    print("- Training Doc2Vec models")
    feature_name = "doc2vec"

    corpus = read(paths.corpus(ds))

    make_dir(paths.feature_model_dir(ds, feature_name))
    make_dir(paths.embedding_dir(ds, feature_name))

    for c in cols:
        model = train_doc2vec_from_file(c)
        emb = model.dv.vectors

        model_save_path = paths.feature_model(ds, feature_name, c)
        model.save(str(model_save_path))

        emb_save_path = paths.embedding(ds, feature_name, c)
        save_npz(emb_save_path, csr_matrix(emb))


def train_tfidf(ds, cols):
    """Trains and saves a TF-IDF model for each text column in the dataset
    Also saves document embeddings learned by the models.
    """
    print("- Training TF-IDF models")
    feature_name = "tfidf"

    corpus = read(paths.corpus(ds))

    make_dir(paths.feature_model_dir(ds, feature_name))
    make_dir(paths.embedding_dir(ds, feature_name))

    for c in cols:
        # tf-idf takes space separated strings
        corpus[c] = corpus[c].apply(lambda l: " ".join(l))
        tfidf = TfidfVectorizer().fit(corpus[c])
        emb = tfidf.transform(corpus[c])

        model_save_path = paths.feature_model(ds, feature_name, c)
        joblib.dump(tfidf, model_save_path)

        emb_save_path = paths.embedding(ds, feature_name, c)
        save_npz(emb_save_path, emb)


def train_lda(ds, cols):
    """Trains and saves an LDA model for each text column in the dataset
    Also saves document embeddings learned by the models.
    """

    def train_from_bow(c):
        """Trains an LDA model from a BoW + Vocab"""
        vocab = Dictionary(corpus[c])
        corpus[c] = corpus[c].apply(vocab.doc2bow)
        lda = lda_model(corpus[c], vocab)
        return lda

    print("- Training LDA models")

    feature_name = "topic"

    corpus = read(paths.corpus(ds))

    make_dir(paths.feature_model_dir(ds, feature_name))
    make_dir(paths.embedding_dir(ds, feature_name))

    for c in cols:
        lda = train_from_bow(c)
        emb = corpus[c].apply(lambda t: sparse2full(lda[t], 100))

        model_save_path = paths.feature_model(ds, feature_name, c)
        lda.save(str(model_save_path))

        emb_save_path = paths.embedding(ds, feature_name, c)
        save_npz(emb_save_path, csr_matrix(list(emb)))


def get_bertoverflow_embeddings(ds, cols, use_gpu=True):
    """Computes BERTOverflow embeddings and saves them"""
    print("- Computing BERTOverflow embeddings")

    feature_name = "bertoverflow"

    corpus = read(paths.corpus(ds, tokenized=False))

    model = bertoverflow_model()

    make_dir(paths.embedding_dir(ds, feature_name))

    device = None
    if not use_gpu:
        device = "cpu"

    for c in cols:
        print(f"-- Computing {c} embeddings with BERTOverflow for {ds}.")
        emb = model.encode(corpus[c], device=device, show_progress_bar=True)

        emb_save_path = paths.embedding(ds, feature_name, c)
        save_npz(emb_save_path, csr_matrix(emb))


def get_mpnet_embeddings(ds, cols, use_gpu=True):
    """Computes MPNet embeddings and saves them"""
    print("- Computing MPNet embeddings")
    feature_name = "mpnet"

    corpus = read(paths.corpus(ds, tokenized=False))

    model = mpnet_model()

    make_dir(paths.embedding_dir(ds, feature_name))

    device = None
    if not use_gpu:
        device = "cpu"

    for c in cols:
        print(f"-- Computing {c} embeddings with MPNet for {ds}.")
        emb = model.encode(corpus[c], device=device, show_progress_bar=True)

        emb_save_path = paths.embedding(ds, feature_name, c)
        save_npz(emb_save_path, csr_matrix(emb))


def train_all_models(ds, feats, cols):
    """Trains all feature models for the given dataset"""
    print(f"Training feature models for {ds}.")
    if "tfidf" in feats:
        train_tfidf(ds, cols)
    if "bm25" in feats:
        train_bm25(ds, cols)
    if "topic" in feats:
        train_lda(ds, cols)
    if "doc2vec" in feats:
        train_doc2vec(ds, cols)
    if "bertoverflow" in feats:
        get_bertoverflow_embeddings(ds, cols)
    if "mpnet" in feats:
        get_mpnet_embeddings(ds, cols)


def main(datasets, feats, cols):
    print(f"Training feature models")
    for ds in datasets:
        train_all_models(ds, feats, cols)


if __name__ == "__main__":
    main(datasets, features, text_columns)
