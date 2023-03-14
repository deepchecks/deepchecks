# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
# This code was copied from https://github.com/AstraZeneca/rexmex
# Under the Apache License Version 2.0, January 2004 http://www.apache.org/licenses/.
# Citation: Rozemberczki, B., Nilsson, S., Hoyt, C. T., & Edwards, G. RexMex (Version 0.1.0) [Computer software].
# https://github.com/AstraZeneca/rexmex
#
"""Used for rec metrics."""
import itertools
from math import log2
from typing import List, Sequence, TypeVar

import numpy as np
from scipy import stats
from sklearn.metrics import dcg_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

X = TypeVar("X")


def reciprocal_rank(relevant_item: X, recommendation: Sequence[X]) -> float:
    """
    Calculate the reciprocal rank (RR) of an item in a ranked list of items.

    Args:
        relevant_item: a target item in the predicted list of items.
        recommendation: An N x 1 sequence of predicted items.
    Returns:
        RR (float): The reciprocal rank of the item.
    """
    for i, item in enumerate(recommendation):
        if item == relevant_item:
            return 1.0 / (i + 1.0)
    return 100


def mean_reciprocal_rank(relevant_items: List, recommendation: List):
    """
    Calculate the mean reciprocal rank (MRR) of items in a ranked list.

    Args:
        relevant_items (array-like): An N x 1 array of relevant items.
        recommendation (array-like): An N x 1 array of ordered items.
    Returns:
        MRR (float): The mean reciprocal rank of the relevant items in a predicted.
    """

    reciprocal_ranks = []
    for item in relevant_items:
        rr = reciprocal_rank(item, recommendation)
        reciprocal_ranks.append(rr)

    return np.mean(reciprocal_ranks)


def rank(relevant_item: X, recommendation: Sequence[X]) -> float:
    """
    Calculate the rank of an item in a ranked list of items.

    Args:
        relevant_item: a target item in the predicted list of items.
        recommendation: An N x 1 sequence of predicted items.
    Returns:
        : The rank of the item.
    """
    for i, item in enumerate(recommendation):
        if item == relevant_item:
            return i + 1.0
    raise ValueError("relevant item did not appear in recommendation")


def mean_rank(relevant_items: Sequence[X], recommendation: Sequence[X]) -> float:
    """
    Calculate the arithmetic mean rank (MR) of items in a ranked list.

    Args:
        relevant_items: An N x 1 sequence of relevant items.
        recommendation: An N x 1 sequence of ordered items.
    Returns:
        : The mean rank of the relevant items in a predicted.
    """
    return np.mean([rank(item, recommendation) for item in relevant_items])


def gmean_rank(relevant_items: Sequence[X], recommendation: Sequence[X]) -> float:
    """
    Calculate the geometric mean rank (GMR) of items in a ranked list.

    Args:
        relevant_items: An N x 1 sequence of relevant items.
        recommendation: An N x 1 sequence of ordered items.
    Returns:
        : The mean reciprocal rank of the relevant items in a predicted.
    """
    return stats.gmean([rank(item, recommendation) for item in relevant_items])


def average_precision_at_k(relevant_items: np.array, recommendation: np.array, k=10):
    """
    Calculate the average precision at k (AP@K) of items in a ranked list.

    Args:
        relevant_items (array-like): An N x 1 array of relevant items.
        recommendation (array-like): An N x 1 array of ordered items.
        k (int): the number of items considered in the predicted list.
    Returns:
        AP@K (float): The average precision @ k of a predicted list.

    `Original <https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py>`_
    """

    if len(recommendation) > k:
        recommendation = recommendation[:k]

    score = 0.0
    hits = 0.0
    for i, item in enumerate(recommendation):
        if item in relevant_items and item not in recommendation[:i]:
            hits += 1.0
            score += hits / (i + 1.0)

    return score / min(len(relevant_items), k)


def mean_average_precision_at_k(relevant_items: List[list], recommendations: List[list], k: int = 10):
    """
    Calculate the mean average precision at k (MAP@K) across predicted lists.
    Each prediction should be paired with a list of relevant items. First predicted list is
    evaluated against the first list of relevant items, and so on.

    Example usage:

    .. code-block:: python

        import numpy as np
        from rexmex.metrics.predicted import mean_average_precision_at_k

        mean_average_precision_at_k(
            relevant_items=np.array(
                [
                    [1,2],
                    [2,3]
                ]
            ),
            predicted=np.array([
                [3,2,1],
                [2,1,3]
            ])
        )
        >>> 0.708333...

    Args:
        relevant_items (array-like): An M x N array of relevant items.
        recommendations (array-like):  An M x N array of recommendation lists.
        k (int): the number of items considered in the predicted list.
    Returns:
        MAP@K (float): The mean average precision @ k across recommendations.
    """

    aps = []
    for items, recommendation in zip(relevant_items, recommendations):
        ap = average_precision_at_k(items, recommendation, k)
        aps.append(ap)

    return np.mean(aps)


def average_recall_at_k(relevant_items: List, recommendation: List, k: int = 10):
    """
    Calculate the average recall at k (AR@K) of items in a ranked list.

    Args:
        relevant_items (array-like): An N x 1 array of relevant items.
        recommendation (array-like):  An N x 1 array of items.
        k (int): the number of items considered in the predicted list.
    Returns:
        AR@K (float): The average precision @ k of a predicted list.
    """
    if len(recommendation) > k:
        recommendation = recommendation[:k]

    num_hits = 0.0
    score = 0.0

    for i, item in enumerate(recommendation):
        if item in relevant_items and item not in recommendation[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(relevant_items)


def mean_average_recall_at_k(relevant_items: List[list], recommendations: List[list], k: int = 10):
    """
    Calculate the mean average recall at k (MAR@K) for a list of recommendations.
    Each recommendation should be paired with a list of relevant items. First recommendation list is
    evaluated against the first list of relevant items, and so on.

    Args:
        relevant_items (array-like): An M x R list where M is the number of recommendation lists,
                                     and R is the number of relevant items.
        recommendations (array-like):  An M x N list where M is the number of recommendation lists and
                                N is the number of recommended items.
        k (int): the number of items considered in the recommendation.
    Returns:
        MAR@K (float): The mean average recall @ k across the recommendations.
    """
    ars = []
    for items, recommendation in zip(relevant_items, recommendations):
        ar = average_recall_at_k(items, recommendation, k)
        ars.append(ar)

    return np.mean(ars)


def hits_at_k(relevant_items: np.array, recommendation: np.array, k=10):
    """
    Calculate the number of hits of relevant items in a ranked list HITS@K.

    Args:
        relevant_items (array-like): An 1 x N array of relevant items.
        recommendation (array-like): An 1 x N array of predicted arrays
        k (int): the number of items considered in the predicted list
    Returns:
        HITS@K (float):  The number of relevant items in the first k items of a prediction.
    """
    if len(recommendation) > k:
        recommendation = recommendation[:k]

    hits = 0.0
    for i, item in enumerate(recommendation):
        if item in relevant_items and item not in recommendation[:i]:
            hits += 1.0

    return hits / len(recommendation)


def spearmans_rho(relevant_items: np.array, recommendation: np.array):
    """
    Calculate the Spearman's rank correlation coefficient (Spearman's rho) between two lists.

    Args:
        relevant_items (array-like): An 1 x N array of items.
        recommendation (array-like):  An 1 x N array of items.
    Returns:
        (float): Spearman's rho.
        p-value (float): two-sided p-value for null hypothesis that both predicted are uncorrelated.
    """
    return stats.spearmanr(relevant_items, recommendation)


def kendall_tau(relevant_items: np.array, recommendation: np.array):
    """
    Calculate the Kendall's tau, measuring the correspondence between two lists.

    Args:
        relevant_items (array-like): An 1 x N array of items.
        recommendation (array-like):  An 1 x N array of items.
    Returns:
        Kendall tau (float): The tau statistic.
        p-value (float): two-sided p-value for null hypothesis that there's no association between the predicted.
    """
    return stats.kendalltau(relevant_items, recommendation)


def intra_list_similarity(recommendations: List[list], items_feature_matrix: np.array):
    """
    Calculate the intra list similarity of recommended items. The items
    are represented by feature vectors, which compared with cosine similarity.
    The predicted consists of item indices, which are used to fetch the item
    features.

    Args:
        recommendations (List[list]): A M x N array of predicted, where M is the number
                               of predicted and N the number of recommended items
        items_feature_matrix (matrix-link): A N x D matrix, where N is the number of items and D the
                                number of features representing one item

    Returns:
        (float): Average intra list similarity across predicted

    `Original <https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py#L232>`_
    """

    intra_list_similarities = []
    for predicted in recommendations:
        predicted_features = items_feature_matrix[predicted]
        similarity = cosine_similarity(predicted_features)
        upper_right = np.triu_indices(similarity.shape[0], k=1)
        avg_similarity = np.mean(similarity[upper_right])
        intra_list_similarities.append(avg_similarity)

    return np.mean(intra_list_similarities)


def personalization(recommendations: List[list]):
    """
    Calculates personalization, a measure of similarity between recommendations.
    A high value indicates that the recommendations are disimillar, or "personalized".

    Args:
        recommendations (List[list]): A M x N array of predicted items, where M is the number
                                    of predicted lists and N the number of items

    Returns:
        (float): personalization

    `Original <https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py#L160>`_
    """

    n_predictions = len(recommendations)

    # map each ranked item to index
    item2ix = {}
    counter = 0
    for prediction in recommendations:
        for item in prediction:
            if item not in item2ix:
                item2ix[item] = counter
                counter += 1

    n_items = len(item2ix.keys())

    # create matrix of predicted x items
    items_matrix = np.zeros((n_predictions, n_items))

    for i, prediction in enumerate(recommendations):
        for item in prediction:
            item_ix = item2ix[item]
            items_matrix[i][item_ix] = 1

    similarity = cosine_similarity(X=items_matrix)
    dim = similarity.shape[0]
    personalization = (similarity.sum() - dim) / (dim * (dim - 1))

    return 1 - personalization


def novelty(recommendations: List[list], item_popularities: dict, num_users: int, k: int = 10):
    """
    Calculates the capacity of the recommender system to to generate novel
    and unexpected results.

    Args:
        recommendations (List[list]): A M x N array of items, where M is the number
                               of predicted lists and N the number of recommended items
        item_popularities (dict): A dict mapping each item in the recommendations to a popularity value.
                                  Popular items have higher values.
        num_users (int): The number of users
        k (int): The number of items considered in each recommendation.

    Returns:
        (float): novelty

    Metric Definition:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.

    `Original <https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py#L14>`_
    """

    epsilon = 1e-10
    all_self_information = []
    for recommendations in recommendations:
        self_information_sum = 0.0
        for i in range(k):
            item = recommendations[i]
            item_pop = item_popularities[item]
            self_information_sum += -log2((item_pop + epsilon) / num_users)

        avg_self_information = self_information_sum / k
        all_self_information.append(avg_self_information)

    return np.mean(all_self_information)


def normalized_distance_based_performance_measure(relevant_items: List, recommendation: List):
    """
    Calculates the Normalized Distance-based Performance Measure (NPDM) between two
    ordered lists. Two matching orderings return 0.0 while two unmatched orderings returns 1.0.

    Args:
        relevant_items (List): List of items
        recommendation (List): The predicted list of items

    Returns:
        NDPM (float): Normalized Distance-based Performance Measure

    Metric Definition:
    Yao, Y. Y. "Measuring retrieval effectiveness based on user preference of documents."
    Journal of the American Society for Information science 46.2 (1995): 133-145.

    Definition from:
    Shani, Guy, and Asela Gunawardana. "Evaluating recommendation systems."
    Recommender systems handbook. Springer, Boston, MA, 2011. 257-297
    """
    assert set(relevant_items) == set(recommendation)

    item_relevant_items_rank = {item: i + 1 for i, item in enumerate(dict.fromkeys(relevant_items))}
    item_predicted_rank = {item: i + 1 for i, item in enumerate(dict.fromkeys(recommendation))}

    items = set(relevant_items)

    item_combinations = itertools.combinations(items, 2)

    C_minus = 0
    C_plus = 0
    C_u = 0

    for item1, item2 in item_combinations:
        item1_relevant_items_rank = item_relevant_items_rank[item1]
        item2_relevant_items_rank = item_relevant_items_rank[item2]

        item1_pred_rank = item_predicted_rank[item1]
        item2_pred_rank = item_predicted_rank[item2]

        C = np.sign(item1_pred_rank - item2_pred_rank) * np.sign(item1_relevant_items_rank - item2_relevant_items_rank)

        C_u += C**2

        if C < 0:
            C_minus += 1
        else:
            C_plus += 1

    C_u0 = C_u - (C_plus + C_minus)

    NDPM = (C_minus + 0.5 * C_u0) / C_u
    return NDPM


def discounted_cumulative_gain(y_true: np.array, y_score: np.array):
    """
    Computes the Discounted Cumulative Gain (DCG), a sum of the true scores ordered
    by the predicted scores, and then penalized by a logarithmic discount based on ordering.

    Args:
        y_true (array-like): An N x M array of ground truth values, where M > 1 for multilabel classification problems.
        y_score (array-like): An N x M array of predicted values, where M > 1 for multilabel classification problems..
    Returns:
        DCG (float): Discounted Cumulative Gain
    """
    return dcg_score(y_true, y_score)


def normalized_discounted_cumulative_gain(y_true: np.array, y_score: np.array):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG), a sum of the true scores ordered
    by the predicted scores, and then penalized by a logarithmic discount based on ordering.
    The score is normalized between [0.0, 1.0]

    Args:
        y_true (array-like): An N x M array of ground truth values, where M > 1 for multilabel classification problems.
        y_score (array-like): An N x M array of predicted values, where M > 1 for multilabel classification problems..
    Returns:
        NDCG (float) : Normalized Discounted Cumulative Gain
    """
    return ndcg_score(y_true, y_score)
