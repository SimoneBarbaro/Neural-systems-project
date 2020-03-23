import numpy as np
from keyword_selection import InvertedIndex, KeywordSelector
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
    Base class to filter corpus based on keyword methods.
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

    def selection_strategy(self, tokens):
        """
        Abstract method where the logic of how to choose keys, this can call different methods of keyword_selector.
        There are many possible solutions and the best must be found with experimentation.
        :param tokens: tokens from which to select the keywords.
        :return: a list of keywords
        """
        raise NotImplementedError

    def filter(self, query):
        """
        Method called to return a subset of document ids, filtered accordingly to a specific query.
        :param query: query to be used.
        :return: a list of document ids.
        """
        tokens = self.preprocessing_fn(query)
        selected_keywords = self.selection_strategy(tokens)
        selected_doc_ids = self.inverted_index.get_doc_id_list(selected_keywords)
        return self.filter_strategy.filter_selection(selected_doc_ids)


class SelectKCorpusFilter(CorpusFilter):
    """
    Simple corpus filter that select at most k keywords based on the keyword scorer where k is given as input.
    """
    def __init__(self, corpus, preprocessing_fn, keyword_selector: KeywordSelector, filter_strategy: FilterStrategy, k):
        super().__init__(corpus, preprocessing_fn, keyword_selector, filter_strategy)
        self.k = k

    def selection_strategy(self, tokens):
        return self.keyword_selector.select_k(tokens, self.k)

