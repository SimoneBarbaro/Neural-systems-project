import numpy as np
import argparse
from parsing import get_part2_datasets, get_doc_id_mapping, SentenceTokenizer, PuctDigitRemoveTokenizer, get_ngrams
from ranking import *
from filter import CorpusFilterBuilder, OrFilterStrategy, AndFilterStrategy
from scoring import discounted_cumulative_gain, ml_score, plot_ml_histograms, plot_ml_curve, \
    rouge_score, print_rouge_score


def get_tokenizer_fn(tokenizer, num_grams):
    if tokenizer == "SentenceTokenizer":
        tokenizer_fn = SentenceTokenizer().tokenize
    elif tokenizer == "PuctDigitRemoveTokenizer":
        tokenizer_fn = PuctDigitRemoveTokenizer().tokenize
    elif tokenizer == "split":
        tokenizer_fn = lambda x: x.split()
    else:
        raise NotImplementedError("Tokenizer not implemented yet")
    if num_grams > 1:
        return lambda x: get_ngrams(tokenizer_fn(x), num_grams)
    return tokenizer_fn


def get_corpus_filter(corpus, tokenizer_fn, filter_name, num_keywords):
    builder = CorpusFilterBuilder(corpus, tokenizer_fn).set_k_keyword_selector(num_keywords)
    if filter_name == "or":
        builder.set_filter_strategy(OrFilterStrategy())
    elif filter_name == "and":
        builder.set_filter_strategy(AndFilterStrategy())
    return builder.build()


def get_ranker(args, corpus):
    if args.ranker == "BM25":
        ranker = Bm25Ranker(corpus, get_tokenizer_fn(args.tokenizer, args.num_grams))
    elif args.ranker == "FastBM25":
        ranker = FastBM25Ranker(corpus, get_tokenizer_fn(args.tokenizer, args.num_grams))
    elif args.ranker == "Sent2Vec":
        ranker = Sent2VecRanker(corpus, split=args.split)
    elif args.ranker == "Bert":
        ranker = BertRanker(corpus, split=args.split)
    elif args.ranker == "BM25Hybrid":
        ranker = Bm25HybridRanker(corpus, get_corpus_filter(corpus, get_tokenizer_fn(args.tokenizer, args.num_grams),
                                                            args.filter, args.filter_num_keywords),
                                  get_tokenizer_fn(args.tokenizer, args.num_grams))
    elif args.ranker == "FastBM25Hybrid":
        ranker = FastBm25HybridRanker(corpus,
                                      get_corpus_filter(corpus, get_tokenizer_fn(args.tokenizer, args.num_grams),
                                                        args.filter, args.filter_num_keywords),
                                      get_tokenizer_fn(args.tokenizer, args.num_grams))
    elif args.ranker == "Sent2VecHybrid":
        ranker = EmbeddingHybridRanker(corpus,
                                       get_corpus_filter(corpus, get_tokenizer_fn(args.tokenizer, args.num_grams),
                                                         args.filter,
                                                         args.filter_num_keywords),
                                       Sent2VecRanker(corpus, split=args.split))
    elif args.ranker == "BertHybrid":
        ranker = EmbeddingHybridRanker(corpus,
                                       get_corpus_filter(corpus, get_tokenizer_fn(args.tokenizer, args.num_grams),
                                                         args.filter,
                                                         args.filter_num_keywords),
                                       BertRanker(corpus, split=args.split))
    else:
        raise NotImplementedError("Ranker not implemented yet")
    return ranker


def test_pairing(args):
    # TODO histogram not working
    results, discussions, pairs = get_part2_datasets(only_pairs=True)
    doc_ids = results.index.unique(0)
    discussions_index_to_id_mapper, discussions_id_to_index_mapper = get_doc_id_mapping(discussions)

    predictions = []
    real_query_ids = []
    for doc_id in doc_ids:
        queries = results.loc[doc_id]["text"].values

        for q in results.loc[doc_id].index:
            real_query_ids += [pairs[(doc_id, q)]]
        corpus = discussions.loc[doc_id]["text"].values
        ranker = get_ranker(args, corpus)
        indexes = ranker.batch_knn_prediction(queries, k=args.k)
        predictions += [[discussions_index_to_id_mapper[res][-1] for res in index] for index in indexes]

    print("M@1 score: {}".format(ml_score(real_query_ids, predictions, L=1)))
    print("M@20 score: {}".format(ml_score(real_query_ids, predictions, L=20)))
    plot_ml_histograms(real_query_ids, predictions)
    plot_ml_curve(real_query_ids, predictions)


def test_retrieval(args):
    results, discussions, scores = get_part2_datasets()
    query_ids = results.index.values
    queries = results["text"].values
    corpus = discussions["text"].values
    ranker = get_ranker(args, corpus)
    discussions_index_to_id_mapper, discussions_id_to_index_mapper = get_doc_id_mapping(discussions)

    indexes = ranker.batch_knn_prediction(queries, k=args.k)
    predictions = [[discussions_index_to_id_mapper[res] for res in index] for index in indexes]

    dcg = np.array(discounted_cumulative_gain(query_ids, predictions, scores, normalize=True))
    dcg_idx = np.argsort(dcg)

    if args.examples > 0:
        print("good examples:")
        for i in range(1, args.examples + 1):
            ind = dcg_idx[-i]
            print("query: {}".format(query_ids[ind]))
            print("dcg: {}".format(dcg[ind]))
            # print(queries[ind])
            print("predictions:")
            for pred in predictions[ind]:
                print("\t{}".format(pred))

                print("\tOur score: {}".format(scores["score"].get(query_ids[ind] + pred, 0)))
        print("bad examples:")
        for i in range(args.examples):
            ind = dcg_idx[i]
            print("query: {}".format(query_ids[ind]))
            print("dcg: {}".format(dcg[ind]))
            # print(queries[ind])
            print("predictions:")
            for pred in predictions[ind]:
                print("\t{}".format(pred))
                print("\tOur score: {}".format(scores["score"].get(query_ids[ind] + pred, 0)))

    print("DCG: {}".format(np.array(discounted_cumulative_gain(query_ids, predictions, scores, normalize=True)).mean()))
    if args.rouge:
        predictions_text = [[discussions.loc[id]['text'] for id in prediction] for prediction in predictions]
        scores = rouge_score(queries, predictions_text)
        print_rouge_score(scores)


def main(args):
    if args.mode == "pairing":
        test_pairing(args)
    else:
        test_retrieval(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="retrieval", help="testing mode", choices=["retrieval", "examples", "pairing"])
    parser.add_argument("--k", type=int, default=20, help="k for knn prediction")
    parser.add_argument("--ranker", type=str, default="FastBM25", help="ranker model",
                        choices=["BM25", "FastBM25", "Sent2Vec", "Bert",
                                 "BM25Hybrid", "FastBM25Hybrid", "Sent2VecHybrid", "BertHybrid"])
    parser.add_argument("--filter", type=str, default=None, help="filtering method, None if not using hybrid ranker",
                        choices=[None, "and", "or"])
    parser.add_argument("--filter_num_keywords", type=int, default=3, help="number of keywords for filtering method")
    parser.add_argument("--tokenizer", type=str, default="SentenceTokenizer",
                        help="tokenizer to use, only some ranker will make use of this parameter",
                        choices=["SentenceTokenizer", "PuctDigitRemoveTokenizer", "split"])
    parser.add_argument("--num_grams", type=int, default=1, help="ngrams for token based rankers")
    parser.add_argument("--split", type=bool, default=False,
                        help="Whether to split document sentences for embedding based rankers")
    parser.add_argument("--rouge", default=False, action='store_true')
    parser.add_argument("--examples", type=int, default=0, help='number of examples to show')

    args = parser.parse_args()

    main(args)
