import numpy as np
import argparse
from parsing import get_part2_datasets, get_doc_id_mapping, SentenceTokenizer
from src.ranking import Bm25Ranker, FastBM25Ranker, Sent2VecRanker, Sent2VecHybridRanker, Bm25HybridRanker, FastBm25HybridRanker
from src.filter import CorpusFilterBuilder, OrFilterStrategy, AndFilterStrategy
from scoring import discounted_cumulative_gain


def get_tokenizer_fn(tokenizer):
    if tokenizer == "SentenceTokenizer":
        return SentenceTokenizer().tokenize
    elif tokenizer == "split":
        return lambda x: x.split()


def get_corpus_filter(corpus, tokenizer_fn, filter_name):
    builder = CorpusFilterBuilder(corpus, tokenizer_fn).set_k_keyword_selector(3)
    if filter_name == "or":
        builder.set_filter_strategy(OrFilterStrategy())
    elif filter_name == "and":
        builder.set_filter_strategy(AndFilterStrategy())
    return builder.build()


def main(args):
    results, discussions, scores = get_part2_datasets()
    query_ids = results.index.values
    queries = results["text"].values
    corpus = discussions["text"].values

    if args.ranker == "BM25":
        ranker = Bm25Ranker(corpus, get_tokenizer_fn(args.tokenizer))
    elif args.ranker == "FastBM25":
        ranker = FastBM25Ranker(corpus, get_tokenizer_fn(args.tokenizer), *args.ranker_args)
    elif args.ranker == "Sec2Sec":
        ranker = Sent2VecRanker(corpus)
    elif args.ranker == "BM25Hybrid":
        ranker = Bm25HybridRanker(corpus, get_corpus_filter(corpus, get_tokenizer_fn(args.tokenizer), args.filter),
                                get_tokenizer_fn(args.tokenizer))
    elif args.ranker == "FastBM25Hybrid":
        ranker = FastBm25HybridRanker(corpus, get_corpus_filter(corpus, get_tokenizer_fn(args.tokenizer), args.filter),
                                    get_tokenizer_fn(args.tokenizer), *args.ranker_args)
    elif args.ranker == "Sec2SecHybrid":
        ranker = Sent2VecHybridRanker(corpus, get_corpus_filter(corpus, get_tokenizer_fn(args.tokenizer), args.filter))

    discussions_index_to_id_mapper, discussions_id_to_index_mapper = get_doc_id_mapping(discussions)

    indexes = ranker.batch_knn_prediction(queries, k=20)
    predictions = [[discussions_index_to_id_mapper[res] for res in index] for index in indexes]

    print(np.array(discounted_cumulative_gain(query_ids, predictions, scores, normalize=True)).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="k for knn prediction")
    parser.add_argument("--ranker", type=str, default="FastBM25Ranker", help="ranker model",
                        choices=["BM25", "FastBM25", "Sent2Vec"])
    parser.add_argument("--filter", type=str, default=None, help="filtering method, None if not using hybrid ranker",
                        choices=[None, "and", "or"])
    parser.add_argument("--ranker_args", nargs="+", help="optionally, args for ranker")
    parser.add_argument("--tokenizer", type=str, default="SentenceTokenizer",
                        help="tokenizer to use, only some ranker will make use of this parameter",
                        choices=["SentenceTokenizer", "split"])

    args = parser.parse_args()

    main(args)
