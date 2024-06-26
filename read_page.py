import csv
import datetime
import json

from bs4 import BeautifulSoup


JS_2004 = ['docs/2004.js']
JS_2008 = ['docs/2008.js', 'docs/2008_2.js','docs/2008_3.js','docs/2008_4.js','docs/2008_5.js','docs/2008_6.js','docs/2008_7.js']
JS_2012 = ['docs/2012.js', 'docs/2012_2.js', 'docs/2012_3.js', 'docs/2012_4.js', 'docs/2012_5.js', 'docs/2012_6.js', ]
JS_2016 = ['docs/2016.js', 'docs/2016_2.js','docs/2016_3.js','docs/2016_4.js','docs/2016_5.js','docs/2016_6.js','docs/2016_7.js',]
JS_2020 = [f'docs/2020_{i}.js' for i in range(7)]
JS_2024 = [f'docs/2024_{i+1}.js' for i in range(5)]

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

def parse_files(filenames, year):

    with open(year + '.csv', 'w', newline='') as f:
        fieldnames = ['date', 'kind', 'home', 'away', 'score', 'max_odd_home', 'mean_odd_home', 'max_odd_draw', 'mean_odd_draw', 'max_odd_away', 'mean_odd_away']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filename in filenames:
            print(f"Reading file {filename}")
            with open(filename) as f:
                data = json.load(f)

            if isinstance(data, str):
                for row in bs_parse(data):
                    writer.writerow(row)
            else:
                # Newer format
                for row in js_parse(data):
                    writer.writerow(row)


def bs_parse(str):
    soup = BeautifulSoup(str, 'html.parser')
    for tr in soup.find_all('tr')[1:]:
        row = {}
        if 'table-dummyrow' in tr['class']:
            continue
        if 'nob-border' in tr['class']:
            date = tr.span['class'][1]
            date = datetime.date.fromtimestamp(int(date[1:-8]))
            kind = next(tr.th.strings)[3:]
            # print(f'date: {date}')
            continue
        row['kind'] = kind
        tds = tr.find_all('td')
        hour = tds[0]['class'][2]
        hour = datetime.datetime.fromtimestamp(int(hour[1:-8]))
        # print(f"time: {hour}")
        row['date'] = hour
        participants = tds[1].get_text().split(' - ')
        # print(f"parts: {participants}")
        row['home'] = participants[0].strip()
        row['away'] = participants[1].strip()
        score = tds[2].get_text()
        if score == 'canc.':
            print(f"Skipping cancelled match {row['home']}-{row['away']}")
            continue
        if score == 'award.':
            print(f"Skipping 'award' match {row['home']}-{row['away']}")
            continue
        row['score'] = score
        if tds[3]['xoid'] == '-' or tds[4]['xoid'] == '-' or tds[5]['xoid'] == '-':
            print(f"Skipping match {row['home']}-{row['away']} with missing odd")
            continue
        odds = parse_odds(tds[3]['xodd'])
        row['max_odd_home'] = odds[0]
        row['mean_odd_home'] = odds[1]
        odds = parse_odds(tds[4]['xodd'])
        row['max_odd_draw'] = odds[0]
        row['mean_odd_draw'] = odds[1]
        odds = parse_odds(tds[5]['xodd'])
        row['max_odd_away'] = odds[0]
        row['mean_odd_away'] = odds[1]
        # print(f"odds: {odds}")
        yield row


def js_parse(str):
    for obj in str['d']['rows']:
        row = {}
        row['date'] = datetime.datetime.fromtimestamp(obj['date-start-timestamp'])
        row['kind'] = obj['tournament-stage-name'][3:]
        row['home'] = obj['home-name']
        row['away'] = obj['away-name']
        row['score'] = obj['postmatchResult'].replace('&nbsp;', '\xa0')
        odds = obj['odds']
        row['max_odd_home'] = odds[0]['maxOdds']
        row['mean_odd_home'] = odds[0]['avgOdds']
        row['max_odd_draw'] = odds[1]['maxOdds']
        row['mean_odd_draw'] = odds[1]['avgOdds']
        row['max_odd_away'] = odds[2]['maxOdds']
        row['mean_odd_away'] = odds[2]['avgOdds']
        yield row




parse_files(JS_2004, '2004')
parse_files(JS_2008, '2008')
parse_files(JS_2012, '2012')
parse_files(JS_2016, '2016')
parse_files(JS_2020, '2020')
parse_files(JS_2024, '2024')
