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
"""Used for rec metrics."""
import itertools
import warnings
from math import log2
from typing import Dict, List, Sequence, TypeVar

import numpy as np
import pandas as pd
from numba import jit, njit
from scipy import stats
from sklearn.metrics import dcg_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

X = TypeVar("X")


# Copied from https://github.com/statisticianinstilettos/recmetrics, Licenced under MIT Licence,
# Copyright (c) 2019 Claire Longo

def prediction_coverage(recommendations: List[list], item_to_index: Dict, unseen_warning: bool = True) -> float:
    """
    Computes the prediction coverage for a list of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    unseen_warn: bool
        when prediction gives any item unseen in catalog:
            (1) ignore the unseen item and warn
            (2) or raise an exception.
    Returns
    ----------
    prediction_coverage:
        The prediction coverage of the recommendations as a percent
        rounded to 2 decimal places
    ----------
    Metric Defintion:
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    catalog = item_to_index.keys()
    unique_items_catalog = set(catalog)
    if len(catalog) != len(unique_items_catalog):
        raise AssertionError("Duplicated items in catalog")

    predicted_flattened = [p for sublist in recommendations for p in sublist]
    unique_items_pred = set(predicted_flattened)

    if not unique_items_pred.issubset(unique_items_catalog):
        if unseen_warning:
            warnings.warn("There are items in predictions but unseen in catalog. "
                          "They are ignored from prediction_coverage calculation")
            unique_items_pred = unique_items_pred.intersection(unique_items_catalog)
        else:
            raise AssertionError("There are items in predictions but unseen in catalog.")

    num_unique_predictions = len(unique_items_pred)
    prediction_coverage = round(num_unique_predictions / (len(catalog) * 1.0) * 100, 2)
    return prediction_coverage

# Cosine similarity code copied from https://github.com/talboger/fastdist, Licenced under MIT Licence,
# Copyright (c) 2020 tal boger

@jit(nopython=True, fastmath=True)
def init_w(w, n):
    """
    :purpose:
    Initialize a weight array consistent of 1s if none is given
    This is called at the start of each function containing a w param
    :params:
    w      : a weight vector, if one was given to the initial function, else None
             NOTE: w MUST be an array of np.float64. so, even if you want a boolean w,
             convert it to np.float64 (using w.astype(np.float64)) before passing it to
             any function
    n      : the desired length of the vector of 1s (often set to len(u))
    :returns:
    w      : an array of 1s with shape (n,) if w is None, else return w un-changed
    """
    if w is None:
        return np.ones(n)
    else:
        return w


@jit(nopython=True, fastmath=True)
def cosine(u, v, w=None):
    """
    :purpose:
    Computes the cosine distance between two 1D arrays
    NOTE: fastdist before v1.1.5 returned cosine similarity (which is 1 - distance)
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    cosine  : float, the cosine distance between u and v
    """
    n = len(u)
    w = init_w(w, n)
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(n):
        num += u[i] * v[i] * w[i]
        u_norm += abs(u[i]) ** 2 * w[i]
        v_norm += abs(v[i]) ** 2 * w[i]

    denom = (u_norm * v_norm) ** (1 / 2)
    return 1 - num / denom


@jit(nopython=True, fastmath=True)
def cosine_vector_to_matrix(u, m):
    """
    :purpose:
    Computes the cosine similarity between a 1D array and rows of a matrix
    :params:
    u      : input vector of shape (n,)
    m      : input matrix of shape (m, n)
    :returns:
    cosine vector  : np.array, of shape (m,) vector containing cosine similarity between u
                     and the rows of m
    (returns an array of shape (100,))
    """
    norm = 0
    for i in range(len(u)):
        norm += abs(u[i]) ** 2
    u_norm = u / norm ** (1 / 2)
    m_norm = np.zeros(m.shape)
    for i in range(m.shape[0]):
        norm = 0
        for j in range(len(m[i])):
            norm += abs(m[i][j]) ** 2
        m_norm[i] = m[i] / norm ** (1 / 2)
    return np.dot(u_norm, m_norm.T)


@jit(nopython=True, fastmath=True)
def cosine_matrix_to_matrix(a, b):
    """
    :purpose:
    Computes the cosine similarity between the rows of two matrices
    :params:
    a, b   : input matrices of shape (m, n) and (k, n)
             the matrices must share a common dimension at index 1
    :returns:
    cosine matrix  : np.array, an (m, k) array of the cosine similarity
                     between the rows of a and b
    (returns an array of shape (10, 100))
    """
    a_norm = np.zeros(a.shape)
    b_norm = np.zeros(b.shape)
    for i in range(a.shape[0]):
        norm = 0
        for j in range(len(a[i])):
            norm += abs(a[i][j]) ** 2
        a_norm[i] = a[i] / norm ** (1 / 2)
    for i in range(b.shape[0]):
        norm = 0
        for j in range(len(b[i])):
            norm += abs(b[i][j]) ** 2
        b_norm[i] = b[i] / norm ** (1 / 2)
    return np.dot(a_norm, b_norm.T)


@jit(nopython=True, fastmath=True)
def cosine_pairwise_distance(a, return_matrix=False):
    """
    :purpose:
    Computes the cosine similarity between the pairwise combinations of the rows of a matrix
    :params:
    a      : input matrix of shape (n, k)
    return_matrix : bool, whether to return the similarity as an (n, n) matrix
                    in which the (i, j) element is the cosine similarity
                    between rows i and j. if true, return the matrix.
                    if false, return a (n choose 2, 1) vector of the
                    similarities
    :returns:
    cosine matrix  : np.array, either an (n, n) matrix if return_matrix=True,
                     or an (n choose 2, 1) array if return_matrix=False
    (returns an array of shape (10, 10))
    """
    n = a.shape[0]
    rows = np.arange(n)
    perm = [(rows[i], rows[j]) for i in range(n) for j in range(i + 1, n)]
    a_norm = np.zeros(a.shape)
    for i in range(n):
        norm = 0
        for j in range(len(a[i])):
            norm += abs(a[i][j]) ** 2
        a_norm[i] = a[i] / norm ** (1 / 2)

    if return_matrix:
        out_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                out_mat[i][j] = np.dot(a_norm[i], a_norm[j])
        out_mat = out_mat + out_mat.T
        np.fill_diagonal(out_mat, 1)
        return out_mat
    else:
        out = np.zeros((len(perm), 1))
        for i in range(len(perm)):
            out[i] = np.dot(a_norm[perm[i][0]], a_norm[perm[i][1]])
        return out


# Metrics implemented internally
# =================================================================================================

def inner_diversity(item_features):
    """Calculate the diversity of a recommendation list based on an internally provided similarity matrix.

    Args:
        item_features (array-like): An N x M array of item features.

    Returns:
        diversity (float): The diversity of the recommendation list.
    """
    # calculate the cosine similarity between the items
    similarity = cosine_pairwise_distance(item_features, return_matrix=True)
    # calculate the diversity
    return 1 - np.mean(similarity[np.triu_indices(len(similarity), k=1)])


def diversity(recommendation: List, item_features: pd.DataFrame) -> float:
    """
    Calculate the diversity of a recommendation list based on an externally provided similarity matrix

    Args:
        recommendation (array-like): An N x 1 array of ordered items.
        item_features (array-like): An N x M array of item features.
    Returns:
        diversity (float): The diversity of the recommendation list.
    """

    # select only the features of the recommended items
    item_features = item_features.loc[recommendation, :]

    return inner_diversity(item_features.values.astype(np.float32))

# The following metrics where copied from https://github.com/AstraZeneca/rexmex
# Under the Apache License Version 2.0, January 2004 http://www.apache.org/licenses/.
# Citation: Rozemberczki, B., Nilsson, S., Hoyt, C. T., & Edwards, G. RexMex (Version 0.1.0) [Computer software].
# https://github.com/AstraZeneca/rexmex
## =================================================================================================


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
    return 0


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
    return None


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

    if len(recommendation) == 0:
        return 0.0
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


def intra_list_similarity(recommendation: List, item_features: pd.DataFrame):
    """
    Calculate the intra list similarity of recommended items. The items
    are represented by feature vectors, which compared with cosine similarity.
    The predicted consists of item indices, which are used to fetch the item
    features.

    Args:
        recommendation (List]): A 1 x N array of predicted, where M is the number
                               of predicted and N the number of recommended items
        item_features (matrix-link): A N x D matrix, where N is the number of items and D the
                                number of features representing one item

    Returns:
        (float): Average intra list similarity across predicted

    `Original <https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py#L232>`_
    """

    intra_list_similarities = []
    predicted_features = item_features.loc[recommendation, :]
    similarity = cosine_similarity(predicted_features)
    upper_right = np.triu_indices(similarity.shape[0], k=1)
    avg_similarity = np.mean(similarity[upper_right])

    return avg_similarity


def personalization(recommendations: List[list]):
    """
    Calculates personalization, a measure of similarity between recommendations.
    A high value indicates that the recommendations are dissimilar, or "personalized".

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


def novelty(recommendation: List, item_popularity: dict, num_users: int, k: int = 10):
    """
    Calculates the capacity of the recommender system to generate novel
    and unexpected results.

    Args:
        recommendation (List): A 1 x N array of items, where M is the number
                               of predicted lists and N the number of recommended items
        item_popularity (dict): A dict mapping each item in the recommendations to a popularity value.
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
    if item_popularity is None:
        raise ValueError("item_popularity must be provided.")

    epsilon = 1e-10
    all_self_information = []
    self_information_sum = 0.0
    for i in range(k):
        item = recommendation[i]
        item_pop = item_popularity[item]
        self_information_sum += -log2((item_pop + epsilon) / num_users)

    avg_self_information = self_information_sum / k

    return avg_self_information


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
