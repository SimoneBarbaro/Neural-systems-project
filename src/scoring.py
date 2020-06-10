import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import dcg_score, ndcg_score
import rouge


def ml_score(real_query_ids, prediction_ids, L):
    """
    Compute M@L score for a prediction.
    :param real_query_ids: true results for a given query.
    :param prediction_ids: predicted results. Array with the first dimension equal to the length of real_query_ids.
    prediction_ids[i, :] contains the sorted predictions by decreasing score.
    :param L: parameter for the metric, number of prediction considered.
    :return: M@L score.
    """
    real_query_ids = np.array(real_query_ids)

    xs = []
    for x in prediction_ids:
        xs.append(np.pad(x, (0, max(L - len(x), 0)), constant_values=-1, )[:L])
    prediction_ids = np.array(xs)

    prediction_ids = np.array(prediction_ids)
    assert prediction_ids.shape[0] == real_query_ids.shape[0]
    assert prediction_ids.shape[1] >= L
    return np.mean(np.any(prediction_ids[:, :L] == real_query_ids[:, np.newaxis], axis=-1))


def plot_ml_curve(real_query_ids, prediction_ids, max_l=20):
    """
    Plot the curve of the M@L scores by changing L.
    :param real_query_ids: true results for a given query.
    :param prediction_ids: predicted results. Array with the first dimension equal to the length of real_query_ids.
    prediction_ids[i, :] contains the sorted predictions by decreasing score.
    :param max_l: maximum L in the plot.
    """
    ml_score_list = []
    for l in range(1, max_l+1):
        ml_score_list.append(ml_score(real_query_ids, prediction_ids, l))
    plt.plot(np.arange(1, max_l+1), ml_score_list)
    plt.xlabel("L")
    plt.ylabel("M@L score")
    plt.title("M@L score curve")
    plt.xticks(np.arange(1, max_l+1))
    plt.show()


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


def rouge_score(dataset, real_query_ids, predictions_ids, aggregator='Avg', metrics=None):
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

    if len(real_query_ids) == 1:
        predictions_ids = [predictions_ids]

    abstract_scores = []
    # title_scores = []

    for id in real_query_ids:
        query_abstracts = dataset[id][1]
        predictions_abstracts = [dataset[i][1] for i in predictions_ids[id]]

        """
        query_titles = dataset[id][0]
        predictions_titles = [dataset[i][0] for i in predictions_ids[id]]
        """

        abstract_score = evaluator.get_scores(query_abstracts, predictions_abstracts)
        # title_score = evaluator.get_scores(query_titles, predictions_titles)

        abstract_scores.append(abstract_score)
        # title_scores.append(title_score)
    return abstract_scores
    # return abstract_scores, title_scores


def print_rouge_score(scores):
    """
    :param scores: dictionary of rouge scores from rouge_score
    :return:
    """
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        p = results['p']
        r = results['r']
        f = results['f']
        print('\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'
              .format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f))
