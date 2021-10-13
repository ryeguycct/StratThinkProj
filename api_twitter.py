import requests, json, yaml



def get_bearer_token():
    with open('config.yml') as f:    
        config_vars = yaml.safe_load(f)
        
    return config_vars['bearer_token']
        


def request_tweet(tweet_id):    
    '''
    source: https://developer.twitter.com/en/docs/twitter-api/tweets/lookup/api-reference
    '''
    
    bearer = get_bearer_token()
    headers = {'Authorization': f'Bearer {bearer}'}
    
    api_base_url = "https://api.twitter.com/2"
    uri = f'{api_base_url}/tweets/{tweet_id}' # a specific tweet
    
    print(f'GET {uri} ...')
    response = requests.get(uri, headers=headers)
    
    print(f'Status Code: {response.status_code}')    
    response_json = response.json() 
    
    return response_json




tweet = request_tweet(1448024326750539778)  

print(tweet)   
    


    





