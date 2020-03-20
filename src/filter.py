import numpy as np
from keyword_selection import InvertedIndex, KeywordSelector


class CorpusFilter:
    """
    Base class to filter corpus based on keyword methods.
    """
    def __init__(self, corpus, preprocessing_fn, keyword_selector: KeywordSelector):
        """
        :param corpus: corpus to be filtered.
        :param preprocessing_fn: function to preprocess and tokenize corpus.
        :param keyword_selector: object that select keywords based on query.
        """
        self.inverted_index = InvertedIndex(corpus, preprocessing_fn)
        self.preprocessing_fn = preprocessing_fn
        self.keyword_selector = keyword_selector

    def selection_strategy(self, tokens):
        """
        Abstract method where the logic of how to choose keys, this can call different methods of keyword_selector.
        There are many possible solutions and the best must be found with experimentation.
        :param tokens: tokens from which to select the keywords.
        :return: a list of keywords
        """
        raise NotImplementedError

    def filter_strategy(self, selected_doc_ids):
        """
        Abstract method used to decide how to filter documents based on which contains what keys.
        There are many possible solutions and the best must be found with experimentation.
        :param selected_doc_ids: array where each element contains an array containing all the ids of the documents
        containing a specific keyword.
        :return: a list of doc_ids filtered by this method.
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
        return self.filter_strategy(selected_doc_ids)
