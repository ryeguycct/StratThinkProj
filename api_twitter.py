# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 22:08:44 2021

@author: ferre
"""

import requests


uri = 'https://api.7thsensepsychics.com/pricing/us'

print(f'GET {uri} ...')

response = requests.get(uri)

print(f'Status Code: {response.status_code}')