import csv
import datetime

import click

FILE = 'fifa_ranking-2021-05-27.csv'

MAP = {
    'Ireland': 'Republic of Ireland',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
}

class FIFARanking(object):

    def __init__(self):
        with open(FILE) as f:
            self.data = list(csv.DictReader(f))
        self.data.sort(reverse=True, key = lambda d: datetime.date.fromisoformat(d['rank_date']))
        self.countries = {}

    def get_ranking(self, country, date = None):
        country = MAP.get(country, country)
        """ Get the ranking of a country at a specific date """
        if date is None:
            date = datetime.date.today()
        elif isinstance(date, str):
            date = datetime.date.fromisoformat(date)
        if country not in self.countries:
            self.countries[country] = list(filter(lambda d: d['country_full'] == country,  self.data))
        for datum in self.countries[country]:
            if datetime.date.fromisoformat(datum['rank_date']) > date:
                # Skip ranks in the future
                continue
            # Return the first known rank before the asked date
            return int(datum['rank'])
        if len(self.countries[country]):
            # We found nothing, yet we have data for the country. Take the oldest entry
            return int(self.countries[country][-1]['rank'])
        print(self.countries[country])
        raise KeyError(country, date)

@click.group()
def cli():
    pass

@cli.command('rank')
@click.argument('country')
@click.option('--date')
def echo_ranking(country, date):
    click.echo(f"On {date}, {country} was {FIFARanking().get_ranking(country, date)}")

if __name__ == "__main__":
    cli()
