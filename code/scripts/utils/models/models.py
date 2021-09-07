import numpy as np
from .bm25 import BM25
from gensim.models import LdaMulticore
from gensim.models.doc2vec import Doc2Vec
from ..consts import n_procs


def doc2vec_model(corpus_path):
    """Defines a Doc2Vec model with predefined parameters"""
    return Doc2Vec(
        corpus_file=str(corpus_path),
        vector_size=30,
        window=15,
        min_count=5,
        workers=n_procs,
        seed=42,
        sample=1e-5,
        negative=1,
        epochs=25,
    )


def bm25_model(corpus):
    """Defines a BM25 model with predefined parameters"""
    return BM25(corpus, k1=0.05, b=0.03)


def lda_model(corpus, vocab):
    """Defines an LDA model with predefined parameters"""
    return LdaMulticore(
        corpus,
        random_state=42,
        id2word=vocab,
        alpha="symmetric",
        eta="auto",
        eval_every=5,
        num_topics=30,
        workers=n_procs,
        minimum_probability=0.0,
    )


def bertoverflow_model():
    """Loads the BERTOverflow model and converts it to a sentence transformer"""
    from sentence_transformers import SentenceTransformer, models

    bertoverflow = models.Transformer("jeniya/BERTOverflow")
    pooling_model = models.Pooling(bertoverflow.get_word_embedding_dimension())
    return SentenceTransformer(modules=[bertoverflow, pooling_model])


def mpnet_model():
    """Loads the MPNet model"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("paraphrase-mpnet-base-v2")
