import requests


def post_image():
    url = 'https://kqua4jnymh.execute-api.us-east-1.amazonaws.com/stage-1/classify'

    # send basic post request
    r = requests.post(url, files={'logo': open('foo.png','rb')})
    return r

r = post_image()
print(r.text)