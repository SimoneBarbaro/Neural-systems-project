import numpy as np
from rank_bm25 import BM25Okapi
import sent2vec
from filter import CorpusFilter


class Ranker:
    """
    Abstract class that rank documents based on a given query.
    The scoring strategy is left not implemented while different utility methods are implemented to avoid replication.
    """
    def score_query(self, query):
        """
        Score corpus based on a given query.
        :param query: input query.
        :return: a list containing the scores for the entire corpus.
        """
        raise NotImplementedError

    def knn_prediction(self, query, k=20):
        """
        Select k documents from the corpus based on the best scores predicted.
        :param query: input query.
        :param k: number of documents to be returned by the prediction.
        :return: a list of at most k document, sorted by decreasing score.
        """
        return np.argsort(-self.score_query(query))[:k]

    def batch_knn_prediction(self, queries, k=20):
        """
        Batched version of knn_prediction that act on multiple queries.
        :param queries: list of input queries.
        :param k: number of documents to be returned by each prediction.
        :return: a batch of prediction of documents.
        """
        result = []
        for query in queries:
            result.append(self.knn_prediction(query, k))
        return result


class Bm25Ranker(Ranker):
    """
    Ranker based on the BM25 model.
    """
    def __init__(self, corpus, tokenizer_fn):
        """
        :param corpus: corpus of documents.
        :param tokenizer_fn: tokenizer function to extract tokens from the documents and the queries.
        """
        self.tokenizer_fn = tokenizer_fn
        tokenized_corpus = [tokenizer_fn(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def score_query(self, query):
        """
        Implementation of the abstract method.
        """
        return self.bm25.get_scores(self.tokenizer_fn(query))


class Sent2VecRanker(Ranker):
    """
    Ranker based on the Sent2Vec model.
    """
    def __init__(self, corpus):
        """
        :param corpus: corpus of documents.
        """
        self.doc_embeddings = []
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model("sent2vec_model.bin")
        for doc in corpus:
            self.doc_embeddings.append(self.model.embed_sentence(doc))
        self.doc_embeddings = np.concatenate(self.doc_embeddings, axis=0)
        # In case that there are zero vectors in the embedding matrix,
        # we only normalize the non-zero vectors in the embedding matrix
        non_zero_indices = np.logical_not(np.all(self.doc_embeddings == 0, axis=1))
        self.doc_embeddings[non_zero_indices] = self.doc_embeddings[non_zero_indices] / np.linalg.norm(
            self.doc_embeddings[non_zero_indices], axis=1, keepdims=True)

    def get_normalized_embedding(self, doc):
        """
        Helper method to create the embedding of a document and normalize it.
        :param doc: input document.
        :return: normalized embedding of the document.
        """
        doc_embedding = self.model.embed_sentence(doc)
        # normalize the query embedding
        if not np.all(doc_embedding == 0):
            normalized_embedding = doc_embedding / np.linalg.norm(doc_embedding, axis=1, keepdims=True)
        else:
            normalized_embedding = doc_embedding
        return normalized_embedding

    def score_query(self, query):
        """
        Implementation of the abstract method.
        """
        query_embedding = self.get_normalized_embedding(query)
        return np.dot(self.doc_embeddings, query_embedding[0])


class Bm25HybridRanker(Ranker):
    """
    Hybrid version of the ranker based on the BM25 model.
    """
    def __init__(self, corpus, corpus_filter: CorpusFilter, tokenizer_fn):
        """
        :param corpus: corpus of documents.
        :param tokenizer_fn: tokenizer function to extract tokens from the documents and the queries.
        :param corpus_filter: filter of corpus based on keywords.
        """
        self.tokenizer_fn = tokenizer_fn
        self.tokenized_corpus = [tokenizer_fn(doc) for doc in corpus]
        self.filter = corpus_filter

    def score_query(self, query):
        """
        Implementation of the abstract method.
        """
        selected_doc_ids = self.filter.filter(query)
        return Bm25Ranker(self.tokenized_corpus[selected_doc_ids], self.tokenizer_fn).score_query(query)


class Sent2VecHybridRanker(Sent2VecRanker):
    def __init__(self, corpus, corpus_filter: CorpusFilter):
        """
        :param corpus: corpus of documents.
        :param corpus_filter: filter of corpus based on keywords.
        """
        super().__init__(corpus)
        self.filter = corpus_filter

    def score_query(self, query):
        """
        Implementation of the abstract method.
        """
        selected_doc_ids = self.filter.filter(query)
        query_embedding = self.get_normalized_embedding(query)
        return np.dot(self.doc_embeddings[selected_doc_ids, :], query_embedding[0])
