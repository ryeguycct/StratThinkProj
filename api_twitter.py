import requests, json


uri = 'https://api.7thsensepsychics.com/pricing/us'

print(f'GET {uri} ...')

response = requests.get(uri)

print(f'Status Code: {response.status_code}')

response_json = response.json()

print(response_json['data']['country'])



# Key yZp9fri3h634hFKskSNH5K6uT
# Secret i241mgTxVisQtFA5iXVcyL2uO8fwohbSnWvAVDH1NbmQgWVdrc
# Bearer Token AAAAAAAAAAAAAAAAAAAAAL2mUgEAAAAA%2FrG0K%2FxOmlRkU9dogGYo%2BTD%2BimU%3DHcHmpKK0NTLRsk6bIP97fQ67tmcQLw9uwDtuzGyw1OTsgladi3