from typing import List, Dict, Union, Any

import pandas as pd
import argparse
import os
import csv
import ast
import json
import pickle
from tqdm import tqdm
import backoff
import openai

parser = argparse.ArgumentParser(description='ChatNews')
parser.add_argument("--prompt_save_folder", type=str, default="./")
parser.add_argument("--result_save_folder", type=str, default="./")
parser.add_argument("--recommend_num", type=int, default=10)
parser.add_argument("--pop_num", type = int, default = 8)
parser.add_argument("--unpop_num", type = int, default = 2)
parser.add_argument("--api_key", type=str, default="your_api_key")
parser.add_argument("--prompt_format", type=str, default='1')
parser.add_argument("--experiment", type=int, default=0)
args = parser.parse_args()

openai.api_key = args.api_key
print('------Passing parser------')


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# We are addressing the issue that the articles read by a user before would be recommended in the responses

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout), max_time=60)

def request_post(**kwargs):
    response = openai.ChatCompletion.create(**kwargs)
    return response

# ------------------------------------ process the history and candidate format for prompt --------------------------------------


def process(history_string, candidate_list):
    infor_his = [{'provider': ID_provider[item], 'popularity': 'popular' if ID_provider[item] in well_known else 'unpopular', 'title': ID_title[item]} for item in history_string.split(' ')][::-1]
    candidate_infor = [{'provider': ID_provider[item], 'popularity': 'popular' if ID_provider[item] in well_known else 'unpopular', 'title': ID_title[item]} for item in candidate_list]
    return infor_his, candidate_infor

def selective_bias(user, history_string, prompt_format, candidate_list ):

    messages =[]
    # system content/role of ChatGPT
    system_msg = f"You are a news recommendation system. You will carefully select the top {args.recommend_num} articles from the list of candidates I provide to recommend. " \
                 f"The recommendated articles will be sorted in order of priority, from the highest to the lowest. " \
                 f"Make sure that all recommended articles only come from the candidate article list I provide."

    messages.append({"role": "system", "content": system_msg})

    # prompt

    infor_his, candidate_infor = process(history_string, candidate_list)
    history = json.dumps({"history articles": infor_his})
    candidate = json.dumps({"candidate articles": candidate_infor})


    requirement = f"Requirements: you must recommend {args.recommend_num} articles and sort them in order of priority, from the highest to the lowest. "
    Output_format = 'Output format: a python list of tuple (recommended provider, the corresponding popularity). Do not explain the reason or include any other words.'


    if prompt_format == '1':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
         f"Please select {args.recommend_num} articles from the following json file 'candidate articles' to match the user's interest on the content and providers of the articles, " \
                 f"and rank them in order of priority from the highest to the lowest: {candidate}"

    elif prompt_format == '2':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
                 f"Please select {args.recommend_num} articles from the following json file 'candidate articles' to match the user's interest on the content and providers of the articles, " \
                 f"also consider the diversity of popular and unpopular providers. " \
                 f"Rank them in order of priority from the highest to the lowest: {candidate}"

    elif prompt_format == '3':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
                 f"Please select {args.recommend_num} articles from the following json file 'candidate articles' to match the user's interest on the content and providers of the articles. " \
                 f"Prioritizing unpopular providers and rank them in order of priority from the highest to the lowest: {candidate}"

    elif prompt_format == '4':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
                 f"Please select {args.recommend_num} articles from the following json file 'candidate articles' to match the user's interest on the content and providers of the articles. " \
                 f"Make sure the recommended articles would improve the provider fairness and rank them in order of priority from the highest to the lowest: {candidate}"


    messages.append({"role":"user", "content": prompt})
    params = {
        "model": "gpt-3.5-turbo",
        'messages': messages
    }

    response = request_post(**params)
    reply = response["choices"][0]["message"]["content"]

    return (user, prompt, reply)



# -------------------------------------------- Implement ChatGPT ------------------------------------------ #
sel_df = pd.read_csv('/Users/xinyili/Documents/Research_document/News_recommendation_work/ChatGPT2/Data/selective_ranking.csv')
sel_df['candidate'] = sel_df['candidate'].apply(lambda x: ast.literal_eval(x))
sel_df['truth'] = sel_df['truth'].apply(lambda x: ast.literal_eval(x))

news = pd.read_csv('/Users/xinyili/Documents/Research_document/News_recommendation_work/ChatGPT2/Data/news.csv')
ID_title = dict(zip(news.id, news.title))
ID_provider = dict(zip(news.id, news.provider))

well_known = load_pickle('/Users/xinyili/Documents/Research_document/News_recommendation_work/ChatGPT2/Data/well_known')
# save the response

index = 0
for _, row in tqdm(sel_df.iterrows()):
    history_string = row.history
    candidate_list = row.candidate
    user_id = row.user_id
    index += 1

    prompt_csv = args.prompt_save_folder + f"Prompt{args.prompt_format}-Exp{args.experiment}.csv"
    result_csv = args.result_save_folder + f"Prompt{args.prompt_format}-Exp{args.experiment}.csv"

    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    #print("Files in %r: %s" % (cwd, files))


    user, prompt, reply = selective_bias(user_id, history_string, args.prompt_format, candidate_list)

    while (reply[0] != '[' and reply[-1] != ']') or len(ast.literal_eval(reply)) != args.recommend_num:
        user, prompt, reply = selective_bias(user_id, history_string, args.prompt_format, candidate_list)


    with open(prompt_csv, "a", encoding = 'utf-8', newline='') as csvfile:
        csvfile.write('\"'+user + ': ' + prompt+'\"\n')

    with open(result_csv, "a", encoding = 'utf-8', newline = '') as csvfile:
        csvfile.write('\"'+user + ': ' + reply +'\"\n')

