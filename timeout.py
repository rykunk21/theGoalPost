import requests

req = requests.get('https://www.sports-reference.com/cbb/schools/')
print(req.headers['Retry-After'])