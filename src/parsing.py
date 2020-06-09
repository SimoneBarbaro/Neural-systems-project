import pandas as pd
import numpy as np
from array import array
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from functools import lru_cache
from nltk.corpus import stopwords

nltk.download('stopwords')


def parse_exemple_file(folder="."):
    """
    Load exemple file.
    :param folder: folder where file resides.
    :return: dataframe parsed from file.
    """
    return pd.read_csv(os.path.join(folder, "dataset.txt"), sep=r"<-SEPARATOR->",
                       header=0, index_col=False, dtype="str", keep_default_na=False)


def get_dataset(df):
    """
    Create example dataset from loaded dataframe.
    :param df: dataframe
    :return: dataset
    """
    return [[title, abstract] for title, abstract in zip(df.title, df.abstract)]


class SentenceTokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = SnowballStemmer("english")
        self.general_stopwords = set(stopwords.words('english'))

    @lru_cache(100000)
    def stem(self, w):
        return self.stemmer.stem(w)

    def tokenize(self, sen, remove_stopwords=False):
        if remove_stopwords:
            sen = " ".join([w for w in sen.split() if w.lower() not in self.general_stopwords])
        wlist = self.tokenizer.tokenize(sen)
        sen = " ".join([self.stem(w.lower()) for w in wlist])
        return sen


def add_unigram(inv_idx, index, doc_unigrams):
    unique_unigrams = {}
    for unigram in doc_unigrams:
        unique_unigrams[unigram] = unique_unigrams.get(unigram, 0) + 1

    for unigram in unique_unigrams:
        if unigram not in inv_idx:
            inv_idx[unigram] = {"doc_indices": array("I", [index]),
                                "term_frequencies": array("I", [unique_unigrams[unigram]])}
        else:
            inv_idx[unigram]["doc_indices"].append(index)
            inv_idx[unigram]["term_frequencies"].append(unique_unigrams[unigram])


def get_doc_id_mapping(corpus_dataframe):
    index_to_id_mapper = {}
    id_to_index_mapper = {}
    index = 0
    for doc_id, row in corpus_dataframe.iterrows():
        index_to_id_mapper[index] = doc_id
        id_to_index_mapper[doc_id] = index
        index += 1


def get_inverted_index_data(corpus, tokenizer_fn):
    index = 0
    inv_idx = {}

    index_to_doc_length_mapper = {}

    for text in corpus:
        fulltext = tokenizer_fn(text.strip())
        fulltext_words = fulltext.split()
        add_unigram(inv_idx, index, fulltext_words)
        index_to_doc_length_mapper[index] = len(fulltext_words)

        index += 1

    for unigram in inv_idx:
        inv_idx[unigram]["doc_indices"] = np.array(inv_idx[unigram]["doc_indices"])
        inv_idx[unigram]["term_frequencies"] = np.array(inv_idx[unigram]["term_frequencies"])

    return {"index_to_doc_length_mapper": index_to_doc_length_mapper,
            "num_of_docs": index,
            "inv_idx": inv_idx}
