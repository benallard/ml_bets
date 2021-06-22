import csv
import datetime

import click

FILE = 'fifa_ranking-2021-05-27.csv'

def read_data():
    with open(FILE) as f:
        data = list(csv.DictReader(f))
    data.sort(reverse=True, key = lambda d: datetime.date.fromisoformat(d['rank_date']))
    return data

def get_ranking(country, date = None, data = None):
    """ Get the ranking of a country at a specific date """
    if data is None:
        data = read_data()
    if date is None:
        date = datetime.date.today()
    else:
        date = datetime.date.fromisoformat(date)
    for datum in data:
        if datetime.date.fromisoformat(datum['rank_date']) > date:
            # Skip ranks in the future
            continue
        if country == datum['country_full']:
            # Return the first known rank before the asked date
            return datum['rank']

@click.group()
def cli():
    pass

@cli.command('rank')
@click.argument('country')
@click.option('--date')
def echo_ranking(country, date):
    click.echo(f"On {date}, {country} was {get_ranking(country, date)}")

if __name__ == "__main__":
    cli()
