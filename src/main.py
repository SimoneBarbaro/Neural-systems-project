import numpy as np
from parsing import get_part2_datasets, get_doc_id_mapping, SentenceTokenizer
from ranking import FastBM25Ranker
from scoring import discounted_cumulative_gain


def main():
    results, discussions, scores = get_part2_datasets()
    query_ids = results.index.values
    queries = results["text"].values
    corpus = discussions["text"].values

    results_index_to_id_mapper, results_id_to_index_mapper = get_doc_id_mapping(results)
    discussions_index_to_id_mapper, discussions_id_to_index_mapper = get_doc_id_mapping(discussions)

    tokenizer_fn = SentenceTokenizer().tokenize

    fast_bm25 = FastBM25Ranker(corpus, tokenizer_fn)

    indexes = fast_bm25.batch_knn_prediction(queries, k=20)
    predictions = [[discussions_index_to_id_mapper[res] for res in index] for index in indexes]

    print(np.array(discounted_cumulative_gain(query_ids, predictions, scores, normalize=True)).mean())


if __name__ == '__main__':
    main()
