import datetime
import json

from bs4 import BeautifulSoup


JS_2004 = 'docs/2004.js'
JS_2008 = 'docs/2008.js'
JS_2012 = 'docs/2012.js'
JS_2016 = 'docs/2016.js'

with open(JS_2004) as f:
    str = json.load(f)
soup = BeautifulSoup(str, 'html.parser')


for tr in soup.find_all('tr'):
    if tr['class'] == 'table-dummyrow':
        continue
    if 'nob-border' in tr['class']:
        date = tr.span['class'][1]
        date = datetime.date.fromtimestamp(int(date[1:-8]))
        print(f'date: {date}')


