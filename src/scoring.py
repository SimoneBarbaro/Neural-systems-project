import numpy as np
from matplotlib import pyplot as plt


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
        xs.append(np.pad(x, (0, L - len(x)), constant_values=-1, ))
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