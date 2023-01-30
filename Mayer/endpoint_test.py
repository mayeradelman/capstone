import requests

url = 'https://kqua4jnymh.execute-api.us-east-1.amazonaws.com/stage-1/classify'

# send basic post request
r = requests.post(url, )
print(r.text)