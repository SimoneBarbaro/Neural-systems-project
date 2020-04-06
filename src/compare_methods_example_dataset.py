"""
File to create comparative plots of different methods on the exemple dataset.
It's not very documented because I don't expect we would use it after visualizing the comparisons.
"""

import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from parsing import parse_exemple_file, get_dataset
import nltk
from nltk import RegexpTokenizer
from ranking import Bm25Ranker, Bm25HybridRanker, Sent2VecHybridRanker, Sent2VecRanker
from filter import CorpusFilterBuilder, AndFilterStrategy
from scoring import ml_score


def preprocessing():
    df = parse_exemple_file()
    ds = get_dataset(df)
    tokenizer = RegexpTokenizer("\w+")

    def tokenizer_fn(doc):
        return doc.split()

    nltk.download('punkt')
    for record in ds:
        record[0] = " ".join(tokenizer.tokenize(record[0].lower()))
        record[1] = " ".join(tokenizer.tokenize(record[1].lower()))
    queries, corpus = zip(*ds)
    return queries, corpus, tokenizer_fn


def save_compare(predictions, labels, name="result.csv"):
    max_l = 20
    real_query_ids = [i for i in range(len(predictions[0]))]
    ml_scores = {}
    for i, prediction_ids in enumerate(predictions):
        ml_score_list = []
        for l in range(1, max_l + 1):
            ml_score_list.append(ml_score(real_query_ids, prediction_ids, l))
        ml_scores[labels[i]] = ml_score_list

    pd.DataFrame(ml_scores, index=[l for l in range(1, max_l + 1)]).transpose().to_csv(name)


def plot_compare(predictions, labels):
    max_l = 20
    real_query_ids = [i for i in range(len(predictions[0]))]
    for i, prediction_ids in enumerate(predictions):
        ml_score_list = []
        for l in range(1, max_l + 1):
            ml_score_list.append(ml_score(real_query_ids, prediction_ids, l))
        plt.plot(np.arange(1, max_l + 1), ml_score_list, label=labels[i])
        plt.xlabel("L")
        plt.ylabel("M@L score")
        plt.title("M@L score curve")
        plt.xticks(np.arange(1, max_l + 1))
    plt.legend(loc='lower right')
    plt.show()


def filter_k_compare(ks, model_builder_fn, hybrid_model_builder_fn, filter_builder_fn, plot=True, save=False,
                     save_name="result"):
    queries, corpus, tokenizer_fn = preprocessing()

    model = model_builder_fn(corpus, tokenizer_fn)
    predictions = [model.batch_knn_prediction(queries)]
    labels = ["sent2vec"]
    for k in ks:
        print("doing {}".format(k))
        corpus_filter = filter_builder_fn(corpus, tokenizer_fn, k)

        model = hybrid_model_builder_fn(corpus, corpus_filter, tokenizer_fn)
        predictions.append(model.batch_knn_prediction(queries))
        labels.append("sent2vec_or_{}".format(k))
    if save:
        save_compare(predictions, labels, name=save_name)
    if plot:
        plot_compare(predictions, labels)


def filter_k_or_compare(model_builder_fn, hybrid_model_builder_fn, plot=True, save=False, save_name="result"):
    filter_k_compare([3, 5, 7, 10], model_builder_fn, hybrid_model_builder_fn,
                     lambda corpus, tokenizer_fn, k: CorpusFilterBuilder(corpus, tokenizer_fn).set_k_keyword_selector(
                         k).build(), plot=plot, save=save, save_name=save_name)


def filter_k_and_compare(model_builder_fn, hybrid_model_builder_fn, plot=True, save=False, save_name="result"):
    filter_k_compare([3, 5, 7, 10], model_builder_fn, hybrid_model_builder_fn,
                     lambda corpus, tokenizer_fn, k: CorpusFilterBuilder(corpus, tokenizer_fn)
                     .set_filter_strategy(AndFilterStrategy())
                     .set_k_keyword_selector(k)
                     .build(), plot=plot, save=save, save_name=save_name)


def simple_compare(plot=True, save=False, save_name="result"):
    queries, corpus, tokenizer_fn = preprocessing()

    bm25 = Bm25Ranker(corpus, tokenizer_fn)
    sent2vec = Sent2VecRanker(corpus)

    corpus_filter_or = CorpusFilterBuilder(corpus, tokenizer_fn).set_k_keyword_selector(3).build()
    corpus_filter_and = CorpusFilterBuilder(corpus, tokenizer_fn) \
        .set_filter_strategy(AndFilterStrategy()) \
        .set_k_keyword_selector(3) \
        .build()
    bm25_hybrid_or = Bm25HybridRanker(corpus, corpus_filter_or, tokenizer_fn)
    sent2vec_hybrid_or = Sent2VecHybridRanker(corpus, corpus_filter_or)
    bm25_hybrid_and = Bm25HybridRanker(corpus, corpus_filter_and, tokenizer_fn)
    sent2vec_hybrid_and = Sent2VecHybridRanker(corpus, corpus_filter_and)

    predictions = [bm25.batch_knn_prediction(queries), bm25_hybrid_or.batch_knn_prediction(queries),
                   bm25_hybrid_and.batch_knn_prediction(queries), sent2vec.batch_knn_prediction(queries),
                   sent2vec_hybrid_or.batch_knn_prediction(queries), sent2vec_hybrid_and.batch_knn_prediction(queries)]
    labels = ["bm25", "bm25_or_hybrid", "bm25_and_hybrid", "sent2vec", "sent2vec_or_hybrid", "sent2vec_and_hybrid"]
    if save:
        save_compare(predictions, labels, name=save_name)
    if plot:
        plot_compare(predictions, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--types', type=str, nargs="+", required=True)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--save_name', type=str, default="result")

    args = parser.parse_args()


    def bm25_model_builder_fn(corpus, tokenizer_fn):
        return Bm25Ranker(corpus, tokenizer_fn)


    def sent2vec_model_builder_fn(corpus, tokenizer_fn):
        return Sent2VecRanker(corpus)


    def bm25_hybrid_model_builder_fn(corpus, corpus_filter, tokenizer_fn):
        return Bm25HybridRanker(corpus, corpus_filter, tokenizer_fn)


    def sent2vec_hybrid_model_builder_fn(corpus, corpus_filter, tokenizer_fn):
        return Sent2VecHybridRanker(corpus, corpus_filter)


    if "bm25_k_or" in args.types:
        print("bm25_k_or")
        filter_k_and_compare(bm25_model_builder_fn, bm25_hybrid_model_builder_fn, plot=args.plot, save=args.save,
                             save_name=args.save_name)
    if "bm25_k_and" in args.types:
        print("bm25_k_and")
        filter_k_or_compare(bm25_model_builder_fn, bm25_hybrid_model_builder_fn, plot=args.plot, save=args.save,
                            save_name=args.save_name)
    if "sent2vec_k_or" in args.types:
        print("sent2vec_k_or")
        filter_k_and_compare(sent2vec_model_builder_fn, sent2vec_hybrid_model_builder_fn, plot=args.plot,
                             save=args.save, save_name=args.save_name)
    if "sent2vec_k_and" in args.types:
        print("sent2vec_k_and")
        filter_k_or_compare(sent2vec_model_builder_fn, sent2vec_hybrid_model_builder_fn, plot=args.plot, save=args.save,
                            save_name=args.save_name)
    if "simple" in args.types:
        simple_compare(plot=args.plot, save=args.save, save_name=args.save_name)
