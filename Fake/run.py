import pandas as pd
import argparse
import os
import csv
import ast
import json
from tqdm import tqdm
import backoff
import openai

parser = argparse.ArgumentParser(description='ChatNews')
parser.add_argument("--prompt_save_folder", type=str, default="./")
parser.add_argument("--result_save_folder", type=str, default="./")
parser.add_argument("--recommend_num", type=int, default=10)
parser.add_argument("--api_key", type=str, default="your_api_key")
parser.add_argument("--prompt_format", type=str, default=None)
parser.add_argument("--experiment", type=int, default=0)
args = parser.parse_args()

openai.api_key = args.api_key
print('------Passing parser------')


# We are addressing the issue that the articles read by a user before would be recommended in the responses

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout), max_time=60)

def request_post(**kwargs):
    response = openai.ChatCompletion.create(**kwargs)
    return response

# ------------------------------------ process the history and candidate format for prompt --------------------------------------

def process1(history_string, candidate_list):
    infor_his = [ID_title[item] for item in history_string.split(' ')][::-1]
    candidate_infor = [ID_title[item] for item in candidate_list]
    return infor_his, candidate_infor

def process2(history_string, candidate_list):
    infor_his = [ID_title[item] for item in history_string.split(' ')][::-1]
    candidate_infor = [{"title": ID_title[item]} for item in candidate_list]
    return  infor_his, candidate_infor

def process3(history_string, candidate_list):
    infor_his = [{"title": ID_title[item]} for item in history_string.split(' ')][::-1]
    candidate_infor = [{"title": ID_title[item]} for item in candidate_list]
    return infor_his, candidate_infor


def ranking(user, history_string, prompt_format, candidate_list ):

    messages =[]
    # system content/role of ChatGPT
    system_msg = f"You are a news recommendation system. You will carefully select the top {args.recommend_num} articles from the list of candidates I provide to recommend. " \
                 f"The recommendated articles will be sorted in order of priority, from the highest to the lowest. " \
                 f"Make sure that all recommended articles only come from the candidate article list I provide."
    messages.append({"role": "system", "content": system_msg})

    # prompt
    if prompt_format == '0':
        infor_his, candidates = process1(history_string, candidate_list)
        history = '\"' + '\", \"'.join(infor_his) + '\"'
        candidate = '\"' + '\", \"'.join(candidates) + '\"'

    elif prompt_format == '1' : # both history and candidates are in lists
        infor_his, candidates = process1(history_string, candidate_list)
        history = '[\"' + '\", \"'.join(infor_his) + '\"]'
        candidate = '[\"' + '\", \"'.join(candidates) + '\"]'

    elif prompt_format == '2': # history in list, candidates in json
        infor_his, candidates = process2(history_string, candidate_list)
        history = '[\"' + '\", \"'.join(infor_his) + '\"]'
        candidate = json.dumps({"candidate articles": candidates})

    elif prompt_format == '3':
        infor_his, candidates = process3(history_string, candidate_list)
        history =  json.dumps({"history articles": infor_his})
        candidate = json.dumps({"candidate articles": candidates})

    requirement = f"Requirements: you must choose {args.recommend_num} articles for recommendation and sort them in order of priority, from the highest to the lowest. "
    Output_format = 'Output format: a python list of title that are exclusively from the provided list. Do not explain the reason or include any other words.'

    if prompt_format == '0':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles (in no particular order): {history}. " \
                 f"From the following candidates, choose the top {args.recommend_num} articles to " \
                 f"recommend to the user and rank them in order of priority from the highest to the lowest: {candidate}."

    elif prompt_format == '1':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles (in no particular order): {history}. " \
                 f"From the candidates listed below, choose the top {args.recommend_num} articles to " \
                 f"recommend to the user and rank them in order of priority from the highest to the lowest. Candidates: {candidate}."

    elif prompt_format == '2':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles (in no particular order): {history}. " \
             f"From the candidates listed in the json file 'candidate articles' below, choose the top {args.recommend_num} articles to " \
             f"recommend to the user and rank them in order of priority from the highest to the lowest: {candidate}."


    elif prompt_format == '3':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
             f"From the candidates listed in the json file 'candidate articles' below, choose the top {args.recommend_num} articles to " \
             f"recommend to the user and rank them in order of priority from the highest to the lowest: {candidate}."




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

index= 0
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


    user, prompt, reply = ranking(user_id, history_string, args.prompt_format, candidate_list)

    while (reply[0] != '[' and reply[-1] != ']') or len(ast.literal_eval(reply)) != args.recommend_num:
        user, prompt, reply = ranking(user_id, history_string, args.prompt_format, candidate_list)

    with open(prompt_csv, "a", encoding = 'utf-8', newline='') as csvfile:
        csvfile.write('\"'+user + ': ' + prompt+'\"\n')

    with open(result_csv, "a", encoding = 'utf-8', newline = '') as csvfile:
        csvfile.write('\"'+user + ': ' + reply +'\"\n')

