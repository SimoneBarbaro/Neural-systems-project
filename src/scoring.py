import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import dcg_score, ndcg_score
import rouge
from collections import defaultdict


def ml_score(real_query_ids, prediction_ids, L):
    """
    Compute M@L score for a prediction.
    :param real_query_ids: true results for a given query.
    :param prediction_ids: predicted results. Array with the first dimension equal to the length of real_query_ids.
    prediction_ids[i, :] contains the sorted predictions by decreasing score.
    :param L: parameter for the metric, number of prediction considered.
    :return: M@L score.
    """
    if isinstance(real_query_ids[0], list):
        length = max(map(len, real_query_ids))
        real_query_ids = np.array([xi + [np.nan] * (length - len(xi)) for xi in real_query_ids])

    else:
        real_query_ids = np.array(real_query_ids).reshape(1, -1)

    xs = []
    if isinstance(prediction_ids[0], list):
        for x in prediction_ids:
            xs.append(np.pad(x, (0, max(L - len(x), 0)), constant_values=-1, )[:L])
        prediction_ids = np.array(xs)
    else:
        prediction_ids = np.pad(prediction_ids, (0, max(L - len(prediction_ids), 0)), constant_values=-1, )[:L]
        prediction_ids = prediction_ids.reshape(1, -1)

    assert prediction_ids.shape[0] == real_query_ids.shape[0]
    assert prediction_ids.shape[1] >= L
    return np.mean(
        [np.sum(np.isin(prediction_ids[i, :L], real_query_ids[i, :])) / min(L, np.sum(~np.isnan(real_query_ids[i, :])))
         for i in
         range(prediction_ids.shape[0])])


def plot_ml_curve(real_query_ids, prediction_ids, max_l=20):
    """
    Plot the curve of the M@L scores by changing L.
    :param real_query_ids: true results for a given query.
    :param prediction_ids: predicted results. Array with the first dimension equal to the length of real_query_ids.
    prediction_ids[i, :] contains the sorted predictions by decreasing score.
    :param max_l: maximum L in the plot.
    """
    ml_score_list = []
    for l in range(1, max_l + 1):
        ml_score_list.append(ml_score(real_query_ids, prediction_ids, l))
    plt.plot(np.arange(1, max_l + 1), ml_score_list)
    plt.xlabel("L")
    plt.ylabel("M@L score")
    plt.title("M@L score curve")
    plt.xticks(np.arange(1, max_l + 1))
    plt.show()
    plt.savefig('pic')

def discounted_cumulative_gain(query_ids, predicted_results, scores, normalize=True):
    dcg = []
    for i, query_id in enumerate(query_ids):
        query_scores = scores.loc[query_id]
        relevant_scores = []
        for result in predicted_results[i]:
            relevant_scores.append(query_scores["score"].get(result, 0))
        if normalize:
            dcg.append(ndcg_score(np.array([relevant_scores]), np.array([range(len(predicted_results[i]), 0, -1)])))
        else:
            dcg.append(dcg_score(np.array([relevant_scores]), np.array([range(len(predicted_results[i]), 0, -1)])))
    return dcg


def plot_ml_histograms(real_query_ids, prediction_ids, max_l=20):
    """
    Plot the histogram of the M@L scores by changing L, showing frequency of correct guesses at each L.
    :param real_query_ids: true results for a given query.
    :param prediction_ids: predicted results. Array with the first dimension equal to the length of real_query_ids.
    prediction_ids[i, :] contains the sorted predictions by decreasing score.
    :param max_l: maximum L in the plot.
    """
    rank_order_list = []
    for idx in range(len(real_query_ids)):
        if real_query_ids[idx] in prediction_ids[idx]:
            rank_order_list.append(np.argwhere(real_query_ids[idx] == prediction_ids[idx])[0, 0])
    plt.hist(rank_order_list, bins=max_l)
    plt.show()


def rouge_score(queries, predictions, aggregator='Avg', metrics=None):
    """
    :param dataset: cleaned dataset containing abstracts.
    :param real_query_ids: true results for a given query.
    :param predictions_ids: predicted results. Array with the first dimension equal to the length of real_query_ids.
    prediction_ids[i, :] contains the sorted predictions by decreasing score.
    :param aggregator: aggregator type for multiple queries and predictions, default 'Avg'
    :param metrics: list rouge metrics to compute, default ['rouge-n', 'rouge-l', 'rouge-w']
    :return: abstract_scores, is a dictionary containing rouge scores.
    """
    assert aggregator in ['Avg', 'Best', 'Individual'], 'Incorrect aggregator.'
    apply_avg = apply_best = False
    if aggregator == 'Avg':
        apply_avg = True
    elif aggregator == 'Best':
        apply_best = True

    if metrics is None:
        metrics = ['rouge-n', 'rouge-l', 'rouge-w']

    evaluator = rouge.Rouge(metrics=metrics, max_n=2, limit_length=False, stemming=False, apply_avg=apply_avg,
                            apply_best=apply_best)

    abstract_scores = []

    for id in range(len(queries)):
        query = queries[id]
        predictions_query = predictions[id]

        abstract_score = evaluator.get_scores(query, predictions_query)

        abstract_scores.append(abstract_score)
    return abstract_scores


def print_rouge_score(scores):
    """
    :param scores: dictionary of rouge scores from rouge_score
    :return:
    """
    if isinstance(scores, dict):
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            p = results['p']
            r = results['r']
            f = results['f']
            print('\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'
                  .format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f))
    else:
        median_scores = defaultdict(lambda: defaultdict(list))
        for score in scores:
            for metric, results in sorted(score.items(), key=lambda x: x[0]):
                median_scores[metric]['p'].append(results['p'])
                median_scores[metric]['r'].append(results['r'])
                median_scores[metric]['f'].append(results['f'])
        for metric, results in sorted(median_scores.items(), key=lambda x: x[0]):
            p = np.median(results['p'])
            r = np.median(results['r'])
            f = np.median(results['f'])
            print('\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'
                  .format('median-' + metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f))
