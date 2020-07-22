import ssl
import urllib.request as urllib
import json
import pandas as pd
import numpy


def read_state_data(state_tla):
    """
    Use a state three letter identifier (lower case) to extrtact state-level data from the
    national json file.
    :param state_tla:
    :return: mdf
    """
    state_tla = state_tla.upper()
    url = "https://raw.githubusercontent.com/covid-19-au/covid-19-au.github.io/prod/src/data/state.json"
    response = urllib.urlopen(url)
    nat_timeseries = json.loads(response.read())
    date = []
    cases = []
    deaths = []
    recovered = []
    tests = []
    for i in nat_timeseries:
        date = numpy.append(date, i)
        cases = numpy.append(cases, nat_timeseries[i][state_tla][0])
        if len(nat_timeseries[i][state_tla]) > 1:
            deaths = numpy.append(deaths, nat_timeseries[i][state_tla][1])
            recovered = numpy.append(recovered, nat_timeseries[i][state_tla][2])
            tests = numpy.append(tests, nat_timeseries[i][state_tla][3])
        else:
            deaths = numpy.append(deaths, 0)
            recovered = numpy.append(recovered, 0)
            tests = numpy.append(tests, 0)

    state_timeseries = {"date": date, "cases": cases, "deaths": deaths, "recovered": recovered, "tests": tests}

    # Create data frame from json
    mdf = pd.DataFrame.from_dict(state_timeseries)
    mdf.set_index(pd.to_datetime(state_timeseries['date'], format='%Y-%m-%d'))

    return mdf


def prepare_data(dataset, look_back=1):
    """
    This function takes a data set and offsets x by look_back returning x and y arrays
    """

    d_x, d_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        d_x.append(a)
        d_y.append(dataset[i + look_back, 0])
    return numpy.array(d_x), numpy.array(d_y)


def get_data(perspective, series_type):
    """
    This function returns and formats data based on the data perspective chosen:
    'global' or 'vic' or 'nsw' or 'qld'
    """

    states = ['vic', 'nsw', 'qld']

    # Read in data
    ssl._create_default_https_context = ssl._create_unverified_context
    if perspective == 'global':
        url = "https://pomber.github.io/covid19/timeseries.json"
        covid_data = pd.read_json('https://pomber.github.io/covid19/timeseries.json')
        for i, row in covid_data.iterrows():
            df = pd.json_normalize(covid_data[covid_data.keys()[i]])
            df = df.assign(country=covid_data.keys()[i])
            if i == 0:
                masterdf = df
            else:
                masterdf = masterdf.append(df)
    else:
        masterdf = read_state_data(perspective)

    # Calculate daily cases and sum to date
    masterdf['date'] = pd.to_datetime(masterdf['date'])
    if perspective == 'global':
        masterdf['daily_counts'] = masterdf['confirmed'].diff()
    elif perspective in states:
        masterdf['daily_counts'] = masterdf['cases'].diff()
    masterdf.loc[masterdf.daily_counts < 0, 'daily_counts'] = 0
    masterdf.loc[masterdf.daily_counts.isnull(), 'daily_counts'] = 0
    for_lstm = masterdf.groupby('date').agg('sum')

    if series_type == 'Case Fatality Rate':
        if perspective == 'global':
            for_lstm["cfr"] = for_lstm["deaths"] / for_lstm["confirmed"]
        elif perspective in states:
            for_lstm["cfr"] = for_lstm["deaths"] / for_lstm["cases"]
        if perspective == 'global':
            for_lstm = for_lstm.drop(['confirmed', 'deaths', 'recovered', 'daily_counts'], axis=1)
        elif perspective in states:
            for_lstm = for_lstm.drop(['cases', 'deaths', 'recovered', 'tests', 'daily_counts'], axis=1)
    elif series_type == 'Infections':
        if perspective == 'global':
            for_lstm = for_lstm.drop(['confirmed', 'deaths', 'recovered'], axis=1)
        elif perspective in states:
            for_lstm = for_lstm.drop(['cases', 'deaths', 'recovered', 'tests'], axis=1)

    return for_lstm
