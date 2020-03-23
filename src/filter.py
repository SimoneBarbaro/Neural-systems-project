import numpy as np
from keyword_selection import InvertedIndex, KeywordSelector, TfidfKeywordScorer, SelectKKeywordSelector
from functools import reduce


class FilterStrategy:
    """
    Interface of the corpus filter based on the results of the keyword selection.
    """
    def filter_selection(self, selected_doc_ids):
        """
        Strategy used to decide how to filter documents based on which contains what keys.
        There are many possible solutions and the best must be found with experimentation.
        :param selected_doc_ids: a list of lists of documents ids associated with different keywords.
        :return: a list of document ids selected according to the specific strategy.
        """
        raise NotImplementedError


class AndFilterStrategy(FilterStrategy):
    """
    Implementation of filter strategy that take the intersection of the documents.
    Said in an other way, it implements the and operation on the keywords boolean filter.
    """

    def filter_selection(self, selected_doc_ids):
        return reduce(np.intersect1d, selected_doc_ids)


class OrFilterStrategy(FilterStrategy):
    """
    Implementation of filter strategy that take the union of the documents.
    Said in an other way, it implements the or operation on the keywords boolean filter.
    """

    def filter_selection(self, selected_doc_ids):
        return reduce(np.union1d, selected_doc_ids)


class CorpusFilter:
    """
    Corpus filter that uses strategies injected in the init method to personalize how the filtering is done.
    """
    def __init__(self, corpus, preprocessing_fn, keyword_selector: KeywordSelector, filter_strategy: FilterStrategy):
        """
        :param corpus: corpus to be filtered.
        :param preprocessing_fn: function to preprocess and tokenize corpus.
        :param keyword_selector: object that select keywords based on query.
        :param filter_strategy: strategy used to decide how to filter documents based on which contains what keys.
        There are many possible solutions and the best must be found with experimentation.
        It takes a list of document ids and return a filtered list.
        """
        self.inverted_index = InvertedIndex(corpus, preprocessing_fn)
        self.preprocessing_fn = preprocessing_fn
        self.keyword_selector = keyword_selector
        self.filter_strategy = filter_strategy

    def filter(self, query):
        """
        Method called to return a subset of document ids, filtered accordingly to a specific query.
        :param query: query to be used.
        :return: a list of document ids.
        """
        tokens = self.preprocessing_fn(query)
        selected_keywords = self.keyword_selector.select_keywords(tokens)
        selected_doc_ids = self.inverted_index.get_doc_id_list(selected_keywords)
        return self.filter_strategy.filter_selection(selected_doc_ids)


class CorpusFilterBuilder:
    def __init__(self, corpus, preprocessing_fn=lambda x: x.split()):
        self.corpus = corpus
        self.preprocessing_fn = preprocessing_fn
        self.filter_strategy = OrFilterStrategy()
        self.inverted_index = InvertedIndex(self.corpus, self.preprocessing_fn)
        self.keyword_scorer = None
        self.keyword_selector = None

    def set_filter_strategy(self, filter_strategy):
        """
        Set the filter strategy to the specified in input.
        Use it only if you want to use a different strategy from the default OrFilterStrategy
        :param filter_strategy: a filter strategy.
        """
        self.filter_strategy = filter_strategy
        return self

    def changed_scorer(self):
        """
        Internal method called to reset the selector when the scorer is set.
        """
        self.keyword_selector = None

    def check_and_set_default_scorer(self):
        """
        Check whether the scorer is set and if not set to default scorer.
        """
        if self.keyword_scorer is None:
            self.set_tf_idf_keyword_scorer()

    def set_tf_idf_keyword_scorer(self):
        """
        Set the keyword scorer to the tf_idf scorer.
        """
        self.keyword_scorer = TfidfKeywordScorer(self.inverted_index, len(self.corpus))
        return self

    def set_k_keyword_selector(self, k):
        """
        Set the keyword selector to the SelectKKeywordSelector selector for a given k.
        If no scorer is selected before calling this method, the if_idf scorer will be selected by default.
        :param k: the maximum number of keywords selected by the selector.
        """
        self.check_and_set_default_scorer()
        self.keyword_selector = SelectKKeywordSelector(self.keyword_scorer, k)
        return self

    def build(self):
        """
        Create the final filter. If the keyword selector is not set it raise an exemption.
        The selector must be manually build for now because a default selector has not been chosen.
        :return: the built filter.
        """
        if self.keyword_selector is None:
            raise Exception("Keyword selector not chosen, cannot build")
        return CorpusFilter(self.corpus, self.preprocessing_fn, self.keyword_selector, self.filter_strategy)
