import requests
import json

json_file = 'api_data.json'
url = 'http://localhost:5000/detect'
with open(json_file) as f:
    data = json.load(f)

server_return = requests.post(url, json=data)
print(server_return)

