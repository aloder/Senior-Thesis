import requests

URL = "http://teamgroupby.web.illinois.edu/add_ingredient.php?ingredient="

ff = 'a'
for i in range(1000):
    r = requests.post(url = URL+(ff*i))
    print(r)