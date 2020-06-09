import numpy as np
from rank_bm25 import BM25Okapi
import sent2vec
from filter import CorpusFilter
from parsing import get_inverted_index_data


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


class FastBM25Ranker(Ranker):
    def __init__(self, corpus, tokenizer_fn, k=1.2, b=0.75, require_tokenize=True):

        # self.tokenizer = SentenceTokenizer()
        self.tokenizer_fn = tokenizer_fn
        self.k = k
        self.b = b
        self.require_tokenize = require_tokenize
        inv_idx_data = get_inverted_index_data(corpus, tokenizer_fn)

        self.inv_idx = inv_idx_data["inv_idx"]
        self.index_to_doc_length_mapper = inv_idx_data["index_to_doc_length_mapper"]
        self.num_of_docs = inv_idx_data["num_of_docs"]
        idx_list = list(self.index_to_doc_length_mapper.keys())
        assert np.min(idx_list) == 0 and np.max(idx_list) == len(idx_list) - 1
        self.doc_lengths = np.array([self.index_to_doc_length_mapper[idx] for idx in range(len(idx_list))])
        self.avg_doc_length = np.mean(self.doc_lengths)

    def score_query(self, query):
        if self.require_tokenize:
            w_list = self.tokenizer_fn(query).split()
        else:
            w_list = query.split()
        unique_words = {}
        for w in w_list:
            unique_words[w] = unique_words.get(w, 0) + 1
        scores = np.zeros(self.num_of_docs, dtype=np.float32)
        for w in unique_words:
            if w not in self.inv_idx:
                continue
            Nw = len(self.inv_idx[w]["doc_indices"])
            doc_length_w = self.doc_lengths[self.inv_idx[w]["doc_indices"]]
            scores[self.inv_idx[w]["doc_indices"]] = scores[self.inv_idx[w]["doc_indices"]] + unique_words[w] * \
                                                     self.inv_idx[w]["term_frequencies"] * (1 + self.k) / (
                                                             self.inv_idx[w]["term_frequencies"] + self.k * (
                                                             1 - self.b + self.b * doc_length_w / self.avg_doc_length)) * np.log(
                1 + (self.num_of_docs - Nw + 0.5) / (Nw + 0.5))
        return scores


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


class HybridRanker(Ranker):
    """
    Abstract Hybrid ranker.
    """

    def __init__(self, corpus, corpus_filter: CorpusFilter):
        """
        :param corpus: corpus of documents.
        :param corpus_filter: filter of corpus based on keywords.
        """
        self.corpus = corpus
        self.filter = corpus_filter

    def score_selected(self, query, selected_doc_ids):
        raise NotImplementedError

    def score_query(self, query):
        """
        Implementation of the abstract method.
        """
        selected_doc_ids = self.filter.filter(query)

        selected_scores = self.score_selected(query, selected_doc_ids)

        scores = np.zeros(len(self.corpus))
        if len(selected_doc_ids) > 0:
            scores[selected_doc_ids] = selected_scores
        return scores


class Bm25HybridRanker(HybridRanker):
    """
    Hybrid version of the ranker based on the BM25 model.
    """

    def __init__(self, corpus, corpus_filter: CorpusFilter, tokenizer_fn):
        """
        :param corpus: corpus of documents.
        :param tokenizer_fn: tokenizer function to extract tokens from the documents and the queries.
        :param corpus_filter: filter of corpus based on keywords.
        """
        super().__init__(corpus, corpus_filter)
        self.tokenizer_fn = tokenizer_fn

    def score_selected(self, query, selected_doc_ids):
        if len(selected_doc_ids) == 0:
            return Bm25Ranker(self.corpus, self.tokenizer_fn).score_query(query)
        return Bm25Ranker(np.array(self.corpus)[selected_doc_ids], self.tokenizer_fn).score_query(query)


class FastBm25HybridRanker(HybridRanker):
    """
    Hybrid version of the ranker based on the BM25 model.
    """

    def __init__(self, corpus, corpus_filter: CorpusFilter, tokenizer_fn, k=1.2, b=0.75, require_tokenize=True):
        """
        :param corpus: corpus of documents.
        :param tokenizer_fn: tokenizer function to extract tokens from the documents and the queries.
        :param corpus_filter: filter of corpus based on keywords.
        """
        super().__init__(corpus, corpus_filter)
        self.tokenizer_fn = tokenizer_fn
        self.k = k
        self.b = b
        self.require_tokenize = require_tokenize

    def score_selected(self, query, selected_doc_ids):
        if len(selected_doc_ids) == 0:
            return FastBM25Ranker(self.corpus, self.tokenizer_fn, k=self.k, b=self.b,
                                  require_tokenize=self.require_tokenize).score_query(query)
        return FastBM25Ranker(np.array(self.corpus)[selected_doc_ids], self.tokenizer_fn, k=self.k, b=self.b,
                              require_tokenize=self.require_tokenize).score_query(query)


class Sent2VecHybridRanker(HybridRanker, Sent2VecRanker):
    def __init__(self, corpus, corpus_filter: CorpusFilter):
        """
        :param corpus: corpus of documents.
        :param corpus_filter: filter of corpus based on keywords.
        """
        Sent2VecRanker.__init__(self, corpus)
        HybridRanker.__init__(self, corpus, corpus_filter)

    def score_selected(self, query, selected_doc_ids):
        query_embedding = self.get_normalized_embedding(query)
        return np.dot(self.doc_embeddings[selected_doc_ids, :], query_embedding[0])

    def score_query(self, query):
        """
        Implementation of the abstract method.
        """
        return HybridRanker.score_query(self, query)
