# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
# #
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
"""Used for rec metrics."""
from typing import List, Union,Tuple
import numpy as np
import pandas as pd
import prince
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler



def recall_k(relevant_items: Union[List[int], np.array, None],
             recommendation: Union[List, np.array, None],
             k: int = 5) -> float:
    """
    Compute the Recall@k score for a recommendation.

    Args:
        relevant_items (Union[List[int], np.array, None]): The set of relevant items for the user.
        recommendation (Union[List, np.array, None]): The recommendation list to be evaluated.
        k (int): The number of top items to consider.

    Returns:
        float: The Recall@k score for the recommendation.
    """
    if relevant_items is None or len(relevant_items) == 0:
        return 0.0
    if recommendation is None or len(recommendation) == 0:
        return 0.0
    relevant_set = set(relevant_items)
    rec_set = set(recommendation[:k])
    hits = list((i in relevant_set) for i in recommendation if i in rec_set)
    score = np.cumsum(hits) / (1 + np.arange(len(hits)))
    return np.sum(score * hits) / min(len(relevant_items),k)


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
    for i,items in enumerate(relevant_items):
        recommendation = recommendations[i]
        recall_i = recall_k(items, recommendation, k)
        recall_k_list.append(recall_i)

    return np.mean(recall_k_list)



def precision_k(relevant_items: Union[List[int], set, None],
             recommendation: Union[List, set, None],
             k: int = 5) -> float:
    """
    Compute the Precision@k score for a set of recommendations.

    Args:
    relevant_items (Union[List[int], set, None]): The set of relevant items.
    recommendation (Union[List, set, None]): The set of recommendations.
    k (int): The number of top items to consider.

    Returns:
    float: The Precision@k score for the set of recommendations.
    """
    if not relevant_items:
        return 0.0
      
    relevant_set = set(relevant_items)
    
    hits = [1.0 if p in relevant_set else 0.0 for i, p in enumerate(recommandation)]
    score = sum([hits[i] / (i+1.0) for i in range(len(hits))])
    
    
    return score / min(len(actual), k)



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
    for i,items in enumerate(relevant_items):
        recommendation = recommendations[i]
        precision_i = precision_k(items, recommendation, k)
        precision_k_list.append(precision_i)

    return np.mean(precision_k_list)



def f1_k(relevant_items: Union[List[int], np.array, None],
         recommendation: Union[List, np.array, None],
         k: int = 5) -> float:
    """
    Compute the F1 score for the recommendations up to the top k items.

    Args:
    relevant_items (Union[List[int], np.array, None]): relevant items for each user.
    recommendation (Union[List, np.array, None]): recommended items for each user..
    k (int): The top k items to consider in the recommendation. Default is 5.

    Returns:
    float:  F1 score up to the top k items.
    """

    precision = precision_k(relevant_items, recommendation, k)
    recall = recall_k(relevant_items, recommendation, k)

    return 2 * (precision * recall) / (precision + recall)


def mean_average_f1_at_k(relevant_items: Union[List[List[int]], np.array],
                          recommendations: Union[List[List[int]], np.array],
                          k: int = 10) -> float:
    """
    Computes the mean average F1-score at k for a list of recommendations and their relevant items.

    Args:
        relevant_items (Union[List[list[int]], np.array]): A list of lists of relevant items.
        recommendations (Union[List[list[int]], np.array]): A list of lists of recommended items.
        k (int): The top-k recommendations to consider. Default is 10.

    Returns:
        float: The mean average F1-score at k.
    """
    f1_list = []
    for i,items in enumerate(relevant_items):
        recommendation = recommendations[i]
        f1_i = f1_k(items, recommendation, k)
        f1_list.append(f1_i)

    return np.mean(f1_list)


def reciprocal_rank(relevant_item,recommendation: Union[List, np.array, None]) -> float:
    """
    Calculates the reciprocal rank for a given recommendation.

    Args:
    - relevant_item: integer representing the relevant item
    - recommendation: list or numpy array of recommended items

    Returns:
    - float: reciprocal rank of the relevant item in the recommendation
    """
    if relevant_item is None or len(recommendation) == 0:
        return 0.0
    for rank, rec_item in enumerate(recommendation):
        if relevant_item == rec_item:
            return 1/(rank+1)
        return 0


def mean_reciprocal_rank(relevant_items: List, recommendations: List[List]) -> float:
    """
    Calculates the mean reciprocal rank (MRR) of the recommendations given for the relevant items.
    
    Args:
        relevant_items (List): A list of relevant items for each user or item.
        recommendations (List[List]): A list of recommended items for each user or item.
    
    Returns:
        float: The mean reciprocal rank (MRR) of the recommendations.
    """
    reciprocal_ranks = [reciprocal_rank(item, recommendation) \
                        for item, recommendation \
                        in zip(relevant_items, recommendations)]

    return np.mean(reciprocal_ranks)

def dcg(y_true : np.ndarray,
        y_pred : np.ndarray,
        k : int) -> float:
    """
    Compute discounted cumulative gain (DCG).
    
    Parameters:
        y_true (numpy.ndarray): Array of true relevance labels.
        y_pred (numpy.ndarray): Array of predicted relevance scores.
        k (int): Number of top items to consider. If None, use all items.
    
    Returns:
        float: DCG score.
    """
    dcg_value = 0
    if k is None:
        k = len(y_true)
    # Sort the true labels and predicted scores by the scores
    idx =  np.argsort(y_pred)[::-1]
    y_true = y_true[idx]
    # Compute the DCG
    log2 = np.log2(np.arange(2, k+2))
    dcg_value = np.sum(y_true / log2)
    return dcg_value

def ndcg_k(y_true : np.ndarray,
           y_pred : np.ndarray,
           k : int) -> float:
    """
    Compute normalized discounted cumulative gain (NDCG).
    
    Parameters:
        y_true (numpy.ndarray): Array of true relevance labels.
        y_pred (numpy.ndarray): Array of predicted relevance scores.
        k (int): Number of top items to consider. If None, use all items.
    
    Returns:
        float: NDCG score.
    """
    if k is None:
        k = len(y_true)
    # Compute the ideal DCG by sorting the true labels in decreasing order
    idx = np.argsort(y_true)[::-1]
    #idx = np.argpartition(y_pred, -k)[-k:]
    ideal_dcg = dcg(y_true[idx], y_true[idx], k)
    # Compute the actual DCG using the predicted scores
    dcg_score = dcg(y_true, y_pred, k)
    # Compute the NDCG
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
    num_items = relevant_items.shape[0]

    # Compute the NDCG@k for each query/user and add to mean_ndcg
    for item, recommendation in zip(relevant_items, recommendations):
        mean_ndcg += ndcg_k(item, recommendation, k)

    # Compute the mean NDCG@k and return
    return mean_ndcg / num_items

def popularity_based_novelty(recommendations: Union[List[list], np.array],
                             items_popularity: Union[dict, None],
                             num_users: int,
                             k: int = 20) -> Tuple[float, float]:
    """
    Calculate the popularity-based novelty metric for a list of recommendations.
    Parameters:
        recommendations (Union[List[list], np.array]): List of recommendations for each user.
        items_popularity (Union[dict, None]): Item popularity scores.
        num_users (int): Total number of users.
        k (int): The number of top items to consider for novelty. Default is 20.

    Returns:
        Tuple[float, float]: The popularity-based novelty score and the normalized score.
    """
    #epsilon = 1e-10
    # Calculate the mean surprisal/self-information of the top K popular head items
    top_information_mean = 0
    most_popular_item_catalog = round(0.05 * len(items_popularity))
    popularity_topk_items = sorted(items_popularity)[-most_popular_item_catalog:]
    for item_ in popularity_topk_items:
        top_information_mean += -np.log2(( item_ + 1e-10) / num_users)
    top_information_mean = top_information_mean / most_popular_item_catalog
    all_self_information = []
    for i,rec in recommendations:
        self_information_sum = 0.0
        for i in range(k):
            item_pop = items_popularity[rec[i]]
            self_information_sum += -np.log2((item_pop + 1e-10) / num_users)

        avg_self_information = self_information_sum / k
        all_self_information.append(avg_self_information)

    # Calculate novelty and normalized novelty
    novelty = np.mean(all_self_information)
    normalized_novelty = novelty / top_information_mean
    return novelty, normalized_novelty

def diversity_score(predictions: List, target_vectors: Union[List, np.array],
                    top_n: int = 10) -> float:
    """
    Compute the diversity score of a set of recommendations.
    Parameters:
    predictions (List): List of predicted item indices for each user.
    target_vectors (Union[List, np.array]): List or array of item feature vectors.
    top_n (int): The number of top recommended items to consider.

    Returns:
        float: Diversity score.
    """
    # If there are no predictions, return 0.
    if len(predictions) == 0:
        return 0.0

    # Compute the cosine similarity matrix between the target vectors.
    sim_matrix = cosine_similarity(target_vectors[:top_n])

    # We only consider the upper or lower matrix triangle and ignore the diagonal as well.
    sim_avg = sim_matrix[np.tril_indices(len(sim_matrix), -1)].mean()

    # Return the diversity score.
    return 1 - sim_avg

def mca(x_categorical: Union[np.array, List[List]],
        n_components: int) -> Tuple:
    '''
    Performs Multiple Correspondence Analysis (MCA) on a matrix of categorical variables.
    Args:
    - X_categorical (Union[np.array, List[List]]): A matrix of categorical variables.
    - n_components (int): The number of dimensions to reduce the data to.

    Returns:
    - Tuple[prince.MCA, np.array]: A tuple containing the MCA object, and the transformed data.
    '''

    mca_model = prince.MCA(n_components=n_components)
    mca_model = mca_model.fit(x_categorical)
    x_numerical = mca_model.transform(x_categorical)
    return mca_model, x_numerical.values


def mean_diversity_score(predictions: List[List],
                         item_features : pd.DataFrame,
                         numerical_cols : Union[List[str],None],
                         categorical_cols : Union[List[str],None],
                         item_id_column : str,
                         top_n: int = 10) -> float:
    """
    This function calculates the mean diversity score of recommendations across multiple users. 
    The diversity score measures the dissimilarity between items recommended to a user,
    and is calculated based on the cosine similarity between the item features.

    Args:
        predictions (List[List]): A list of recommendation lists for multiple users..

        top_n (int): The number of top items to consider when computing the diversity score.

    Returns:
        float: The mean diversity score across multiple users.
    """
    num_features_dict = item_features[[item_id_column]+numerical_cols]\
                        .set_index(item_id_column)\
                        .apply(list, axis=1).to_dict()
    similarity_score = 0.0
    ## If the user has categorical features
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
        concat_dict = ({key: num_features_dict[key] + update_categorical_features[key]
                       for key in num_features_dict.keys() & update_categorical_features.keys()}
                    )

        data = [concat_dict[key] for key in concat_dict]
        scaler.fit(data)
        for rec in predictions:
            target_vectors = (np.array([[concat_dict.get(key) for key in rec]])
                              .reshape(len(rec), -1)[:top_n]
                            )
            similarity_score += diversity_score(rec, scaler.transform(target_vectors), top_n)
        return similarity_score/len(predictions)

    ## If there is no categorical features
    for rec in predictions:
        target_vectors = (np.array([[num_features_dict.get(key) for key in rec]])
                          .reshape(len(rec), -1)[:top_n]
                        )
        similarity_score += diversity_score(rec, target_vectors, top_n)

    return similarity_score/len(predictions)

def coverage(train_items: Union[List, np.array],
              recommandations: Union[List[List], np.array]) -> float:
    '''
    Compute the percent of items in the training set that are recommended by the model.

    Args:
    train_items (Union[List, np.array]): List of item IDs in the training set.
    recommandations (Union[List[List], np.array]): List of recommended items for each user.

    Returns:
    float: Percent of items in the training set that are recommended by the model on the test set.
    '''
    if len(train_items) == 0:
        raise ValueError("Argument 'train_items' cannot be empty.")
    # Flatten the list of recommendations into a single list
    flat_recommendations = [item for sublist in recommandations for item in sublist]
    # Create a set from the flat list to get unique items
    unique_recommendations_set = set(flat_recommendations)
    # Convert the set back to a list to get the desired output
    unique_recommendations_list = list(unique_recommendations_set)
    return len(unique_recommendations_list) / len(train_items)
