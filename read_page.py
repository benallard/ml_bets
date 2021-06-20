import csv
import datetime
import json

from bs4 import BeautifulSoup


JS_2004 = 'docs/2004.js'
JS_2008 = 'docs/2008.js'
JS_2012 = 'docs/2012.js'
JS_2016 = 'docs/2016.js'


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

def parse_file(filename, year):

    with open(filename) as f:
        str = json.load(f)
    soup = BeautifulSoup(str, 'html.parser')

    with open(year + '.csv', 'w', newline='') as f:
        fieldnames = ['date', 'home', 'away', 'score', 'max_odd_home', 'mean_odd_home', 'max_odd_none', 'mean_odd_none', 'max_odd_away', 'mean_odd_away']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for tr in soup.find_all('tr')[1:]:
            row = {}
            if 'table-dummyrow' in tr['class']:
                continue
            if 'nob-border' in tr['class']:
                date = tr.span['class'][1]
                date = datetime.date.fromtimestamp(int(date[1:-8]))
                #print(f'date: {date}')
                continue
            tds = tr.find_all('td')
            hour = tds[0]['class'][2]
            hour = datetime.datetime.fromtimestamp(int(hour[1:-8]))
            #print(f"time: {hour}")
            row['date'] = hour
            participants = tds[1].get_text().split(' - ')
            #print(f"parts: {participants}")
            row['home'] = participants[0]
            row['away'] = participants[1]
            score = tds[2].get_text()
            row['score'] = score
            odds = [parse_odds(tds[3]['xodd']),
                parse_odds(tds[4]['xodd']),
                parse_odds(tds[5]['xodd'])]
            row['max_odd_home'] = odds[0][1]
            row['mean_odd_home'] = odds[0][1]
            row['max_odd_none'] = odds[1][0]
            row['mean_odd_none'] = odds[1][1]
            row['max_odd_away'] = odds[2][0]
            row['mean_odd_away'] = odds[2][1]
            #print(f"odds: {odds}")
            writer.writerow(row)

parse_file(JS_2004, '2004')
parse_file(JS_2008, '2008')
parse_file(JS_2012, '2012')
parse_file(JS_2016, '2016')
