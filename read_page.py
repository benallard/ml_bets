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

def parse_odds(str):
    def _parse_odd(str):
        res = ''
        for c in str:
            if c in '0123456789':
                res += c
            elif c == 'z':
                res += '.'
            else:
                res += {
                    'a': '1',
                    'x': '2',
                    'c': '3',
                    't': '4',
                    'e': '5',
                    'o': '6',
                    'p': '7'}[c]
        return float(res)
    str = str.split('f')
    return (_parse_odd(str[0]), _parse_odd(str[1]))

for tr in soup.find_all('tr')[1:]:
    if 'table-dummyrow' in tr['class']:
        continue
    if 'nob-border' in tr['class']:
        date = tr.span['class'][1]
        date = datetime.date.fromtimestamp(int(date[1:-8]))
        print(f'date: {date}')
        continue
    tds = tr.find_all('td')
    hour = tds[0]['class'][2]
    hour = datetime.datetime.fromtimestamp(int(hour[1:-8]))
    print(f"time: {hour}")
    participants = tds[1].get_text().split(' - ')
    print(f"parts: {participants}")
    score = tds[2].get_text()
    odds = [parse_odds(tds[3]['xodd']),
        parse_odds(tds[4]['xodd']),
        parse_odds(tds[5]['xodd'])]
    print(f"odds: {odds}")


