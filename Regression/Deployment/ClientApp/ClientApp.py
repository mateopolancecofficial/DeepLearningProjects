import requests
import json


# defining the api-endpoint
API_ENDPOINT = "http://127.0.0.1:5000/model/score"

# data to be sent to api
data = {"data": {'x0': [9.353670, 10.457191],
                 'x1': [-5.681740, -8.771499],
                 'x2': [1.739091, 1.500788],
                 'x3': [26.117712, 41.984114]},
        "model": 0}

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, data=json.dumps(data), headers=headers)

# extracting response content
prediction = r.text
print("Target value is:%s" % prediction)
