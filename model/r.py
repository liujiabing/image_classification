#!/usr/bin/env python
#_*_coding:utf-8_*_

import requests
url = 'http://118.24.13.245:5000/predict'
files = {'file': open('dataset/4/201807292355.jpg', 'rb')}
r = requests.post(url, files=files, timeout=3)
print r.json()
