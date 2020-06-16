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


def get_part2_datasets(only_pairs=False):
    results = pd.read_csv("../dataset/results.csv")
    discussions = pd.read_csv("../dataset/discussions.csv")
    results = results.set_index(["doc_id", "result_id"]).sort_index()
    discussions = discussions.set_index(["doc_id", "discussion_id"]).sort_index()
    if not only_pairs:
        scores = pd.read_csv("../dataset/scores.csv").drop_duplicates()
        scores = scores.set_index(["doc_id_result", "result_id", "doc_id_discussion", "discussion_id"]).sort_index()
        result_ids = scores.index.to_frame(False)["doc_id_result"].unique()
        discussion_ids = scores.index.to_frame(False)["doc_id_discussion"].unique()
        results = results.loc[result_ids]
        discussions = discussions.loc[discussion_ids]
        return results, discussions, scores
    else:
        pairs = pd.read_csv("../dataset/pairs.csv")

        #results = results.loc[pairs[["doc_id", "result_id"]].drop_duplicates().values].sort_index()
        pairs_map = {}
        for i, row in pairs.iterrows():
            key = row["doc_id"], row["result_id"]
            if pairs_map.get(key, None) is None:
                pairs_map[key] = []
            pairs_map[key].append(row["discussion_id"])
        results = results[results.index.isin(pairs_map.keys())]
        return results, discussions, pairs_map


class SentenceTokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = SnowballStemmer("english")
        self.general_stopwords = set(stopwords.words('english'))

    @lru_cache(100000)
    def stem(self, w):
        return self.stemmer.stem(w)

    def tokenize(self, sen, remove_stopwords=True):
        if remove_stopwords:
            sen = " ".join([w for w in sen.split() if w.lower() not in self.general_stopwords])
        wlist = self.tokenizer.tokenize(sen)
        sen = " ".join([self.stem(w.lower()) for w in wlist])
        return sen.split()


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
    return index_to_id_mapper, id_to_index_mapper


def get_inverted_index_data(corpus, tokenizer_fn):
    index = 0
    inv_idx = {}

    index_to_doc_length_mapper = {}

    for text in corpus:
        fulltext_words = tokenizer_fn(text.strip())
        add_unigram(inv_idx, index, fulltext_words)
        index_to_doc_length_mapper[index] = len(fulltext_words)

        index += 1

    for unigram in inv_idx:
        inv_idx[unigram]["doc_indices"] = np.array(inv_idx[unigram]["doc_indices"])
        inv_idx[unigram]["term_frequencies"] = np.array(inv_idx[unigram]["term_frequencies"])

    return {"index_to_doc_length_mapper": index_to_doc_length_mapper,
            "num_of_docs": index,
            "inv_idx": inv_idx}
