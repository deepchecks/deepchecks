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
from typing import List, Union, Tuple
import numpy as np
import pandas as pd
import itertools
import prince
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings


def recall_k(relevant_items: Union[List[int], np.array, None],
             recommendations: Union[List, np.array, None],
             k: int = 10) -> float:
    """
    Compute the Recall@k score for a recommendations.

    Args:
        relevant_items (Union[List[int], np.array, None]): The set of relevant items for the user.
        recommendations (Union[List, np.array, None]): The recommendations list to be evaluated.
        k (int): The number of top items to consider.

    Returns:
        float: The Recall@k score for the recommendations.
    """
    if relevant_items is None or len(relevant_items) == 0:
        return 0.0
    if recommendations is None or len(recommendations) == 0:
        return 0.0
    relevant_set = set(relevant_items)
    rec_set = set(recommendations[:k])
    hits = [1 for p in rec_set if p in relevant_set]
    score = np.sum(hits)
    return score / min(len(relevant_items), k)


def mean_average_recall_at_k(relevant_items: Union[List[List[int]], np.array],
                             recommendations: Union[List[List[int]], np.array],
                             k: int = 10) -> float:
    """
    Compute the Mean Average Recall (MAR) at k score for a set of recommendations.

    Args:
    relevant_items (Union[List[list[int]], np.array]): set of relevant items for each user.
    recommendations (Union[List[list[int]], np.array]): set of recommendations for each user/query.
    k (int) : The number of top items to consider.

    Returns:
    float: The Mean Average Recall (MAR) at k score for the set of recommendations.
    """
    recall_k_list = []
    for i, items in enumerate(relevant_items):
        recall_i = recall_k(items, recommendations[i], k)
        recall_k_list.append(recall_i)

    return np.mean(recall_k_list)


def precision_k(relevant_items: Union[List[int], None],
                recommendations: Union[List, None],
                k: int = 10) -> float:
    """
    Compute the Precision@k score for a set of recommendations.

    Args:
    relevant_items (Union[List[int], set, None]): The set of relevant items.
    recommendations (Union[List, set, None]): The set of recommendations.
    k (int): The number of top items to consider.

    Returns:
    float: The Precision@k score for the set of recommendations.
    """
    if relevant_items is None or len(relevant_items) == 0:
        return 0.0
    if recommendations is None or len(recommendations) == 0:
        return 0.0

    num_hits = 0.0
    score = 0.0
    # Speed up
    relevant_set = set(relevant_items)
    for i, p in enumerate(itertools.islice(recommendations, k)):
        if p in relevant_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(relevant_items), k)


def mean_average_precision_at_k(relevant_items: Union[List[List[int]], np.array],
                                recommendations: Union[List[List[int]], np.array],
                                k: int = 10) -> float:
    """
    Compute the mean average precision@k for a set of recommendations.

    Args:
    relevant_items (Union[List[list[int]], np.array]): relevant items for each user.
    recommendations (Union[List[list[int]], np.array]): recommendations for each user.
    k (int): An integer specifying the number of top items to consider. Default is 10.

    Returns:
    float: The mean average precision@k score for the set of recommendations.
    """
    precision_k_list = []
    for i, items in enumerate(relevant_items):
        precision_i = precision_k(items, recommendations[i], k)
        precision_k_list.append(precision_i)

    return np.mean(precision_k_list)


def f1_k(relevant_items: Union[List[int], np.array, None],
         recommendations: Union[List, np.array, None],
         k: int = 10) -> float:
    """
    Compute the F1 score for the recommendations up to the top k items.

    Args:
    relevant_items (Union[List[int], np.array, None]): relevant items for each user.
    recommendations (Union[List, np.array, None]): recommended items for each user..
    k (int): The top k items to consider in the recommendations. Default is 5.

    Returns:
    float:  F1 score up to the top k items.
    """
    precision = precision_k(relevant_items, recommendations, k)
    recall = recall_k(relevant_items, recommendations, k)

    if precision == 0 and recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def mean_average_f1_at_k(relevant_items: Union[List[List[int]], np.array],
                         recommendations: Union[List[List[int]], np.array],
                         k: int = 10) -> float:
    """
    Compute the mean average F1-score at k for a list of recommendations and their relevant items.

    Args:
        relevant_items (Union[List[list[int]], np.array]): A list of lists of relevant items.
        recommendations (Union[List[list[int]], np.array]): A list of lists of recommended items.
        k (int): The top-k recommendations to consider. Default is 10.

    Returns:
        float: The mean average F1-score at k.
    """
    avg_precision = mean_average_precision_at_k(relevant_items, recommendations, k)
    avg_recall = mean_average_recall_at_k(relevant_items, recommendations, k)

    if avg_precision == 0 and avg_recall == 0:
        return 0.0

    return 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)


def reciprocal_rank(relevant_items: Union[List, np.array, None],
                    recommendations: Union[List, np.array, None]) -> float:
    """
    Compute the reciprocal rank for a given recommendations.

    Args:
    - relevant_item: list  representing the relevant item
    - recommendations: list or numpy array of recommended items

    Returns:
    - float: reciprocal rank of the relevant item in the recommendations
    """
    def issue_reciprocal_rank_warning():
        warnings.warn("Reciprocal rank is a reranking metric; missing relevant items may impact results.", UserWarning)

    if relevant_items is None or len(recommendations) == 0:
        return 0.0
    for rank, rec_item in enumerate(recommendations):
        if relevant_items[0] == rec_item:
            return 1 / (rank + 1)

        # Raise a warning only if reciprocal rank is 0 and issue it once
        if rank == 0:
            issue_reciprocal_rank_warning()
            break

    return 0


def mean_reciprocal_rank(relevant_items: List[List], recommendations: List[List]) -> float:
    """
    Compute the mean reciprocal rank (MRR) of the recommendations given for the relevant items.

    Args:
        relevant_items (List): A list of list of relevant items for each user or item.
        recommendations (List[List]): A list of recommended items for each user or item.

    Returns:
        float: The mean reciprocal rank (MRR) of the recommendations.
    """
    reciprocal_ranks = [reciprocal_rank(item, recommendation)
                        for item, recommendation
                        in zip(relevant_items, recommendations)]

    return np.mean(reciprocal_ranks)


def dcg(relevant_items, recommendations, k=None):
    """
    Compute discounted cumulative gain (DCG).

    Parameters:
        relevant_items (numpy.ndarray): Array of true relevance labels.
        recommendations (numpy.ndarray): Array of predicted relevance scores.
        k (int): Number of top items to consider. If None, use all items.

    Returns:
        float: DCG score.
    """
    if isinstance(relevant_items, list):
        relevant_items = np.array(relevant_items)
    if isinstance(recommendations, list):
        recommendations = np.array(recommendations)

    num_relevant_items = len(relevant_items)
    num_recommendations = len(recommendations)

    if k is None or k > num_recommendations:
        k = num_recommendations

    # Ensure k doesn't exceed the bounds of relevant_items
    k = min(k, num_relevant_items)

    # Sort the indices of recommendations while considering only the first len(relevant_items) indices
    sorted_indices = np.argsort(recommendations[:num_relevant_items])[::-1]
    sorted_relevant_items = relevant_items[sorted_indices]

    # Compute the discounted cumulative gain
    log2 = np.log2(np.arange(2, k + 2))
    dcg_value = np.sum(sorted_relevant_items[:k] / log2)
    return dcg_value


def ndcg_k(relevant_items : np.ndarray,
           y_pred : np.ndarray,
           k : int) -> float:
    """
    Compute normalized discounted cumulative gain (NDCG).

    Parameters:
        relevant_items (numpy.ndarray): Array of true relevance labels.
        recommendations (numpy.ndarray): Array of predicted relevance scores.
        k (int): Number of top items to consider. If None, use all items.

    Returns:
        float: NDCG score.
    """
    if isinstance(relevant_items, List):
        relevant_items = np.array(relevant_items)
    if isinstance(y_pred, List):
        y_pred = np.array(y_pred)

    # Compute the ideal discounted cumulative gain by sorting the true labels in decreasing order
    indexes = np.argsort(relevant_items)[::-1]

    ideal_dcg = dcg(relevant_items[indexes], relevant_items[indexes], k)
    # Compute the actual discounted cumulative gain using the predicted scores
    dcg_score = dcg(relevant_items, y_pred, k)
    # Compute the normalized discounted cumulative gain
    ndcg_score = dcg_score / ideal_dcg if ideal_dcg > 0 else 0
    return ndcg_score


def mean_average_ndcg_k(relevant_items: Union[List, np.array],
                        recommendations: Union[List, np.array],
                        k: Union[int, None] = None) -> float:
    """
    Compute the mean average Normalized Discounted Cumulative Gain (NDCG@k).

    Parameters:
    relevant_items (Union[List,np.array]): relevant items for each query/user.
    recommendations (Union[List,np.array]): recommended items for each query/user.
    k (Union[int, None]): The number of top items to consider.
    Returns:
    float: Mean Average NDCG@k score.
    """
    mean_ndcg = 0

    # Compute the NDCG@k for each query/user
    for item, recommandation in zip(relevant_items, recommendations):
        mean_ndcg += ndcg_k(item, recommandation, k)

    # Compute the mean NDCG@k and return
    return mean_ndcg / len(relevant_items)


def popularity_based_novelty(recommendations: Union[List[list], np.array],
                             items_popularity: Union[dict, None],
                             num_users: int,
                             k: int = 20) -> float:
    """
    Calculate the popularity-based novelty metric for a list of recommendations.

    Parameters:
        recommendations (Union[List[list], np.array]): A list or array containing recommendation lists for each user.
        items_popularity (Union[dict, None]): A dictionary containing the popularity of items. None if not available.
        num_users (int): The total number of users.
        k (int, optional): The number of top items to consider from each recommendation list. Default is 20.

    Returns:
        Tuple[float, float]: The popularity-based novelty score and the normalized score.
    """
    top_information_mean = 0
    for item_ in items_popularity:
        top_information_mean += -np.log2((item_ + 1e-10) / num_users)
    top_information_mean = top_information_mean / len(items_popularity)
    all_self_information = []
    for rec in recommendations:
        self_information_sum = 0.0
        for i in range(min(k, len(rec))):
            item_pop = items_popularity[rec[i]]
            self_information_sum += -np.log2((item_pop + 1e-10) / num_users)

        avg_self_information = self_information_sum / k
        all_self_information.append(avg_self_information)

    # Calculate novelty and normalized novelty
    novelty = np.mean(all_self_information)
    # normalized_novelty = novelty / top_information_mean
    return novelty


def diversity_score(recommandations: List,
                    item_features: Union[List, np.array],
                    top_n: int = 10) -> float:
    """
    Compute the diversity score of a set of recommendations.

    Parameters:
    recommandations (List): List of predicted item indices for each user.
    item_features (Union[List, np.array]): List or array of item feature vectors.
    top_n (int): The number of top recommended items to consider.

    Returns:
        float: Diversity score.
    """
    # If there are no predictions, return 0.
    if len(recommandations) == 0:
        return 0.0

    # Compute the cosine similarity matrix between the target vectors.
    sim_matrix = cosine_similarity(item_features[:top_n])

    # We only consider the upper or lower matrix triangle and ignore the diagonal as well.
    sim_avg = sim_matrix[np.tril_indices(len(sim_matrix), -1)].mean()

    # Return the diversity score.
    return 1 - sim_avg


def mca(x_categorical: Union[np.array, List[List]],
        n_components: int) -> Tuple:
    """
    Perform Multiple Correspondence Analysis (MCA) on a matrix of categorical variables.

    Args:
    - X_categorical (Union[np.array, List[List]]): A matrix of categorical variables.
    - n_components (int): The number of dimensions to reduce the data to.

    Returns:
    - Tuple[prince.MCA, np.array]: A tuple containing the MCA object, and the transformed data.
    """
    mca_model = prince.MCA(n_components=n_components)
    mca_model = mca_model.fit(x_categorical)
    x_numerical = mca_model.transform(x_categorical)
    return mca_model, x_numerical.values


def mean_diversity_score(predictions: List[List],
                         item_features : pd.DataFrame,
                         numerical_cols : Union[List[str], None],
                         categorical_cols : Union[List[str], None],
                         item_id_column : str,
                         top_n: int = 10) -> float:
    """
    Compute the mean diversity score of recommendations across multiple users.

    The diversity score measures the dissimilarity between items recommended to a user,
    and is calculated based on the cosine similarity between the item features.

    Args:
        predictions (List[List]): A list of recommendations lists for multiple users..

        top_n (int): The number of top items to consider when computing the diversity score.

    Returns:
        float: The mean diversity score across multiple users.
    """
    num_features_dict = item_features[[item_id_column]+numerical_cols]\
        .set_index(item_id_column)\
        .apply(list, axis=1).to_dict()
    similarity_score = 0.0
    # If the user has categorical features
    eps = 1e-10
    scaler = StandardScaler()
    if not isinstance(categorical_cols, type(None)):
        print("Applying MCA ; compute associations between variables")
        cat_features_dict = item_features[[item_id_column]+categorical_cols]\
            .set_index(item_id_column)\
            .apply(list, axis=1).to_dict()
        _, cat_embeddings = mca(item_features[categorical_cols],
                                n_components=round((item_features[categorical_cols].shape[1]+eps)/2))

        update_categorical_features = dict(zip(cat_features_dict.keys(), cat_embeddings.tolist()))

        concat_dict = {
            key: num_features_dict[key] + update_categorical_features[key]
            for key in num_features_dict.keys() & update_categorical_features.keys()
        }

        data = [concat_dict[key] for key in concat_dict]
        scaler.fit(data)
        for rec in predictions:
            target_vectors = (
                np.array([[concat_dict.get(key) for key in rec]])
                .reshape(len(rec), -1)[:top_n]
            )

            similarity_score += diversity_score(rec, scaler.transform(target_vectors), top_n)
        return similarity_score/len(predictions)

    # If there is no categorical features
    for rec in predictions:
        target_vectors = (
            np.array([[num_features_dict.get(key) for key in rec]])
            .reshape(len(rec), -1)[:top_n]
        )
        similarity_score += diversity_score(rec, target_vectors, top_n)

    return similarity_score/len(predictions)


def coverage(train_items: Union[List, np.array],
             recommandations: Union[List[List], np.array]) -> float:
    """
    Compute the percent of items in the training set that are recommended by the model.

    Args:
    train_items (Union[List, np.array]): List of item IDs in the training set.
    recommandations (Union[List[List], np.array]): List of recommended items for each user.

    Returns:
    float: Percent of items in the training set that are recommended by the model on the test set.
    """
    if len(train_items) == 0:
        raise ValueError("Argument 'train_items' cannot be empty.")
    # Flatten the list of recommendations into a single list
    flat_recommendations = [item for sublist in recommandations for item in sublist]
    # Create a set from the flat list to get unique items
    unique_recommendations_set = set(flat_recommendations)
    # Convert the set back to a list to get the desired output
    unique_recommendations_list = list(unique_recommendations_set)
    return len(unique_recommendations_list) / len(train_items)
