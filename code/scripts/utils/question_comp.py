import numpy as np
from scipy.sparse import load_npz
from scipy.special import rel_entr
from scipy.spatial.distance import cdist
from gensim.matutils import jaccard_distance
from sklearn.metrics.pairwise import cosine_similarity

from . import paths as paths
from . import read, save
from .models.bm25 import BM25


class QuestionComp:
    """Class for comparing sets of questions using different similarity measures"""

    def __init__(self, dataset, model_name, column):
        self.dataset = dataset
        self.model_name = model_name
        self.column = column
        self.embeddings = None
        self.load_embedding()

    def load_embedding(self):
        """Loads the document representations for a dataset/model/column
        using an appropriate method
        """
        if self.model_name == "bm25":
            emb = BM25.load(
                paths.feature_model(self.dataset, self.model_name, self.column)
            )
        elif self.model_name == "jaccard":
            # creates sets of words
            emb = read(paths.corpus(self.dataset))[[self.column, "corpus_index"]]
            emb = emb.set_index("corpus_index")[self.column].apply(set)
        else:
            emb = load_npz(paths.embedding(self.dataset, self.model_name, self.column))

            # lda embeddings are not arrays
            if self.model_name == "lda":
                emb = emb.toarray()
        self.embeddings = emb

    def topic_sim(self, indexes, others):
        """Calculates topic similarity between sets of documents represented by indexes
        Each index indicates the position of the document in the corpus
        and serves to select its representation in the embedding matrix
        """
        embedding = self.embeddings[indexes]
        other_embeddings = self.embeddings[others]
        res = cdist(embedding, other_embeddings, metric="jensenshannon")
        # converting distance to similarity
        res = np.negative(res)
        res = np.add(1, res)
        return res

    def bm25_sim(self, indexes, others):
        """Calculates BM25 scores between sets of documents represented by indexes
        Each index indicates the position of the document in the corpus
        and serves to select its representation in the corpus representation
        in the BM25 class
        """

        def compare(i):
            return self.embeddings.compare_documents(i, others)

        return [compare(i) for i in indexes]

    def jac_sim(self, indexes, others):
        """Calculates Jaccard similarities between sets of documents represented by indexes
        Each index indicates the position of the document in the corpus
        and serves to select the corresponding set of words
        """

        def compare(i):
            embedding = self.embeddings.loc[i]
            other_embeddings = self.embeddings.loc[others]
            return other_embeddings.apply(
                lambda t: 1.0 - jaccard_distance(t, embedding)
            ).to_list()

        return [compare(i) for i in indexes]

    def cosine_sim(self, indexes, others):
        """Calculates cosine similarity between sets of documents represented by indexes
        Each index indicates the position of the document in the corpus
        and serves to select its representation in the embedding matrix
        This function is used for TF-IDF, Doc2Vec, MPNet and BertOverflow similarities
        """
        embedding = self.embeddings[indexes]
        other_embeddings = self.embeddings[others]
        return cosine_similarity(embedding, other_embeddings)

    def compare(self, indexes, others):
        """Compares sets of documents represented by indexes using the
        appropriate similarity function
        """
        # allows for passing a single index instead of a list of indexes
        single_index = type(indexes) in (int, np.int64)
        single_comp = type(others) in (int, np.int64)
        if single_index:
            indexes = [indexes]
        if single_comp:
            others = [others]

        if self.model_name == "jaccard":
            scores = self.jac_sim(indexes, others)
        elif self.model_name == "bm25":
            scores = self.bm25_sim(indexes, others)
        elif self.model_name == "lda":
            scores = self.topic_sim(indexes, others)
        else:
            scores = self.cosine_sim(indexes, others)

        # return a single value/array instead of a matrix
        if single_index:
            scores = scores[0]
            if single_comp:
                scores = scores[0]

        return scores
