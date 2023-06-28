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
"""Module containing code for LLM based topic extraction. EXPERIMENTAL."""
import openai
import pandas as pd
from langchain import PromptTemplate

template1 = """Your purpose is to extract topics for a list of queries.

Example Format:
QUERY_LIST:
first query here
second query here
next query here
last query here

The format of your output should be of json structure.
The json will include strings of up to four words, describing the main topics that are common to many of the samples.
maximum of 20 strings.
no further content should be added after the json output.

Input for evaluation:
QUERY_LIST:
{query_list}

"""

TOPIC_EXTRACTOR_RESPONSE_TEMPLATE = PromptTemplate(input_variables=["query_list"],
                                                   template=template1)


template2 = """Your purpose is to extract unique identifying topics.
Each line will contain a list of topics. For each line you should choose one topic (or define a new one)
that categorizes best that list of topics, while keeping it most unique from the others.
So the target is to find what are the most accurate distinct descriptions for each line of topics.

Example Format:
TOPICS_LISTS:
topics_group_0: ['first_topic_1', 'second_topic_1', 'third_topic_1', 'fourth_topic_1']
topics_group_1: ['first_topic_2', 'second_topic_2', 'third_topic_2', 'fourth_topic_2', 'fifth_topic_2']
topics_group_2: ['first_topic_3']

The format of your output should be of json structure.
For each topic_gropu the json should include strings of up to two words describing it.
no further content should be added after the json output.

Input for evaluation:
TOPICS_LIST:
{topics_list}

"""

UNIQUE_TOPICS_EXTRACTOR_RESPONSE_TEMPLATE = PromptTemplate(input_variables=["topics_list"],
                                                           template=template2)

# MODEL PARAMS
MODEL = "gpt-3.5-turbo"
SYSTEM_PROMPT = """You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. 
Your output should be of json structure, with no further output"""


# CLUSTER  PARAMS
MAX_NUM_TO_RUN_ON = 200
RANDOM_STATE = 42
NUM_CLUSTERS = 5
# range of numbers to sample for building topics
MAX_SAMPLE_SIZE_FROM_CLUSTER = 15
MIN_SAMPLE_SIZE_FROM_CLUSTER = 5


def get_chat_completion_content(user_prompt, system_prompt=SYSTEM_PROMPT):
    # get chat response
    for i in range(10):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=600,
                n=1,
                stop=None,
                temperature=0.5
            )
            break
        except:
            continue

    return response['choices'][0]['message']['content']


def run_topics_extraction_prompts(prompts_with_cluster_queries):
    per_cluster_topics = []
    for user_prompt in prompts_with_cluster_queries:
        # call openai for each prompt (according to number of clusters)
        response = get_chat_completion_content(user_prompt)
        per_cluster_topics.append(eval(response)['topics'])
    return per_cluster_topics


def build_prompt_with_all_topic_groups(per_cluster_topics):
    topics_strings = ['topics_group_{}: {}'.format(i, str(single_cluster_topics)) \
                  for i,single_cluster_topics in enumerate(per_cluster_topics)]
    all_topic_groups_string ="\n".join(topics_strings)
    get_uniques_user_prompt = UNIQUE_TOPICS_EXTRACTOR_RESPONSE_TEMPLATE.format(topics_list=all_topic_groups_string)
    return get_uniques_user_prompt


def run_prompt_for_overall_topics(get_uniques_user_prompt):
    for i in range(10):
        result = get_chat_completion_content(get_uniques_user_prompt)
        try:
            overall_topics_dict = eval(result)
            break
        except:
            continue
    return overall_topics_dict


def build_topic_extraction_prompts(text_and_clusters, cluster_col_name):
    smallest_cluster_size = text_and_clusters.groupby(cluster_col_name).size().min()
    SAMPLE_SIZE_FROM_CLUSTER = min(smallest_cluster_size, MAX_SAMPLE_SIZE_FROM_CLUSTER)

    # sample some from cluster
    sampled_queries = text_and_clusters.groupby(cluster_col_name).sample(
        SAMPLE_SIZE_FROM_CLUSTER, random_state=RANDOM_STATE)

    # join queries in each cluster
    queries_per_cluster = sampled_queries.groupby(cluster_col_name).apply(lambda x: '\n'.join(x["text"]))

    # build prompt for each cluster
    prompts_with_cluster_queries = queries_per_cluster.apply(\
                    lambda x: TOPIC_EXTRACTOR_RESPONSE_TEMPLATE.format(query_list=x))

    return prompts_with_cluster_queries


def get_topics(texts, cluster_labels):
    cluster_col_name = 'cluster_id'
    text_and_clusters = pd.concat([pd.Series(texts, name="text"),
                                   pd.Series(cluster_labels, name=cluster_col_name)], axis=1)
    prompts_with_cluster_queries = build_topic_extraction_prompts(text_and_clusters, cluster_col_name)
    per_cluster_topics = run_topics_extraction_prompts(prompts_with_cluster_queries)
    get_uniques_prompt = build_prompt_with_all_topic_groups(per_cluster_topics)
    overall_topics_dict = run_prompt_for_overall_topics(get_uniques_prompt)
    return text_and_clusters[cluster_col_name].apply(lambda x: overall_topics_dict.get("topics_group_{}".format(x),
                                                                                       None))
