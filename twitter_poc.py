# main.py
#Query Twitter via search term(s) and output CSV file
#Author:Ryan Foster

import os

# tweepy is a library/api available to use
import tweepy as tw
import pandas as pd

from config import *

# As Victor discussed earlier, create a config file and put your keys, secrets, tokens in it as variables
# rename the ALL CAPS variables below to match the names you use
auth = tw.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tw.API(auth, wait_on_rate_limit=True)

# pick a search term to search tweets by
search_terms = "#bitcoin OR bitcoin OR BTC -airdrop filter:verified"
#for now just use todays date, because there are restrictions on how far back we can search TBC
search_from_date = "2021-10-16"

# just starting by searching for 1 tweet, just to see it working, you can change where it says .items()
tweets = tw.Cursor(api.search_tweets, q=search_terms, lang="en", until=search_from_date).items(10)

# print out tweet text to check
# for tweet in tweets:
#    print(tweet.text)

# create a list of tweets, where each tweet contains selected data
tweet_data_list = [[tweet.text, tweet.created_at, tweet.user] for tweet in tweets]

# create a dataframe that we can then use to visualize or output to a file
tweet_so_sweet = pd.DataFrame(data=tweet_data_list, columns=['tweet', 'time', 'user'])

#print(tweet_so_sweet)

# save your collected tweets to a csv file
tweet_so_sweet.to_csv('tweetsforme.csv', index=False)