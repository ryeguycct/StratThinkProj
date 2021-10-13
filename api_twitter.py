import requests, json, yaml


uri = 'https://api.7thsensepsychics.com/pricing/us'

print(f'GET {uri} ...')

response = requests.get(uri)

print(f'Status Code: {response.status_code}')

response_json = response.json()

print(response_json['data']['country'])

with open('config.yml') as f:    
    config_vars = yaml.safe_load(f)
    
print(config_vars['api_key'])    

