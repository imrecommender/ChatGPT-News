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
parser.add_argument("--pop_num", type = int, default = 5)
parser.add_argument("--unpop_num", type = int, default = 5)
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


def process(history_string):
    infor_his = [{'provider': ID_provider[item], 'popularity': 'popular' if ID_provider[item] in well_known else 'unpopular', 'title': ID_title[item]} for item in history_string.split(' ')][::-1]
    return infor_his

def generative_bias(user, history_string, prompt_format ):

    messages =[]
    # system content/role of ChatGPT
    system_msg = f"You are an intelligent news recommendation system designed to assist users in discovering articles based on their preferences. " \
                 f"Through careful analysis of the user's reading behavior, you will recommend {args.recommend_num} news articles. " \
                 f"Your recommendation strategy aims to strike a balance between popular and unpopular providers, ensuring a diverse range of sources. " \
                 f"The pre-defined popular providers are ['cnn', 'people', 'usatoday', 'yahoo', 'abcnews', 'insider', 'apnews', 'nbcnews', 'cbsnews', 'foxnews', 'nytimes', 'washingtonpost', 'bbc', 'wonderwall', 'reuters', 'businessinsider', 'thedodo', 'thehill', 'etonline', 'espn', 'eonline', 'forbes', 'cbssports', 'sportingnews', 'bbc.co.uk', 'bloomberg', 'time', 'youtube', 'today', 'latimes', 'theguardian', 'usmagazine', 'cnbc', 'wsj', 'marketwatch', 'bestlifeonline', 'motortrend', 'newsweek', 'usnews','caranddriver', 'npr', 'rd', 'tasteofhome', 'dailymail.co.uk', 'bnnbloomberg.ca', 'yardbarker', 'familyhandyman', 'larrybrownsports', 'eatthis', 'hollywoodreporter', 'huffpost', 'elle', 'hotnewhiphop', 'washingtonexaminer', 'wikipedia','blog.cheapism', 'popsugar', 'profootballtalk.nbcsports', 'politico', 'sportsnaut', 'pinterest', 'motor1', 'harpersbazaar', 'vogue', 'nationalgeographic', 'bostonglobe', 'roadandtrack', 'nesn', 'nfl', 'businessinsider.in', 'Facebook', 'cbc.ca', 'ew', 'seattletimes', 'patch', 'loveproperty', 'rollingstone', 'popularmechanics', 'townandcountrymag', 'stacker', 'variety', 'womenshealthmag', 'france24', 'thedailybeast', 'pbs', 'travelandleisure', 'delish', 'thehealthy', 'independent.co.uk', 'vanityfair', 'travelpulse', 'aljazeera', 'menshealth', 'golfdigest', 'nbcdfw', 'jalopnik', 'britannica', 'healthline', 'gobankingrates', 'imdb'], and others outside of the list are considered as unpopular. "\
                 f"The recommendated articles will be sorted in order of priority, from the highest to the lowest. "
    messages.append({"role": "system", "content": system_msg})

    # prompt

    infor_his = process(history_string)
    history = json.dumps({"history articles": infor_his})


    requirement = f"Requirements: you must recommend {args.recommend_num} articles and sort them in order of priority, from the highest to the lowest. "
    Output_format = 'Output format: a python list of tuple (recommended provider, the corresponding popularity). Do not explain the reason or include any other words.'


    if prompt_format == '1':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
         f"Please recommend {args.recommend_num} articles that match the user's interest while balancing the number of popular and unpopular providers" \

    elif prompt_format == '2':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
                 f"Please recommend {args.recommend_num} articles that match the user's interest while {args.pop_num} are from popular providers and {args.unpop_num} are from unpopular providers." \

    elif prompt_format == '3':
        prompt = f"{requirement}{Output_format} \nHere is a list of 100 popular providers: {well_known}, and here is a list of 68 unpopular providers: {unpop}\nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
                 f"Please recommend {args.recommend_num} articles that match the user's interest while {args.pop_num} are from popular providers and {args.unpop_num} are from unpopular providers." \

    elif prompt_format == '0':
        prompt = f"{requirement}{Output_format} \nThe user has interacted with the following articles in the json file 'history articles': {history}. " \
                 f"Please recommend {args.recommend_num} articles that match the user's interest." \

    messages.append({"role":"user", "content": prompt})
    params = {
        "model": "gpt-3.5-turbo",
        'messages': messages
    }

    response = request_post(**params)
    reply = response["choices"][0]["message"]["content"]

    return (user, prompt, reply)



# -------------------------------------------- Implement ChatGPT ------------------------------------------ #
gen_df = pd.read_csv('/Users/xinyili/Documents/Research_document/News_recommendation_work/ChatGPT2/Data/generative.csv')
news = pd.read_csv('/Users/xinyili/Documents/Research_document/News_recommendation_work/ChatGPT2/Data/news.csv')
ID_title = dict(zip(news.id, news.title))
ID_provider = dict(zip(news.id, news.provider))
well_known = load_pickle('/Users/xinyili/Documents/Research_document/News_recommendation_work/ChatGPT2/Data/well_known')
unpop = load_pickle('/Users/xinyili/Documents/Research_document/News_recommendation_work/ChatGPT2/Data/unpop')

index = 0
for _, row in tqdm(gen_df.iterrows()):
    history_string = row.history
    user_id = row.user_id
    index += 1

    prompt_csv = args.prompt_save_folder + f"Prompt{args.prompt_format}-Exp{args.experiment}.csv"
    result_csv = args.result_save_folder + f"Prompt{args.prompt_format}-Exp{args.experiment}.csv"

    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    #print("Files in %r: %s" % (cwd, files))


    user, prompt, reply = generative_bias(user_id, history_string, args.prompt_format)

    while (reply[0] != '[' and reply[-1] != ']') or len(ast.literal_eval(reply)) != args.recommend_num:
        user, prompt, reply = generative_bias(user_id, history_string, args.prompt_format)

    with open(prompt_csv, "a", encoding = 'utf-8', newline='') as csvfile:
        csvfile.write('\"'+user + ': ' + prompt+'\"\n')

    with open(result_csv, "a", encoding = 'utf-8', newline = '') as csvfile:
        csvfile.write('\"'+user + ': ' + reply +'\"\n')

