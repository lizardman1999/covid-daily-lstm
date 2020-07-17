# Required modules and packages
import numpy
import ssl
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow
import urllib.request as urllib
import json

masterdf = pd.DataFrame

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


def get_data(perspective):
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
    if perspective == 'global':
        for_lstm = for_lstm.drop(['confirmed', 'deaths', 'recovered'], axis=1)
    elif perspective in states:
        for_lstm = for_lstm.drop(['cases', 'deaths', 'recovered', 'tests'], axis=1)

    return for_lstm


def run_daily_stats(perspective
                    , train_sample_size=1):
    """
    This program reads covid cases and calculates daily increase before applying a lstm forecast
    """

    if train_sample_size != 1:
        do_test = True
    else:
        do_test = False

    # Fix random seed for reproducibility
    numpy.random.seed(42)
    tensorflow.random.set_seed(42)

    # normalize the dataset
    for_lstm = get_data(perspective)
    ds2 = for_lstm.values
    ds2 = ds2.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds2 = scaler.fit_transform(ds2)

    # split into train and test sets
    tr_size = int(len(ds2) * train_sample_size)
    tr, tst = ds2[0:tr_size, :], ds2[tr_size:len(ds2), :]

    # reshape into X=t and Y=t+1
    look_back = 10
    tr_x, tr_y = prepare_data(tr, look_back)
    if do_test:
        tst_x, tst_y = prepare_data(tst, look_back)

    # reshape input to be [samples, time steps, features]
    tr_x = numpy.reshape(tr_x, (tr_x.shape[0], 1, tr_x.shape[1]))
    if do_test:
        tst_x = numpy.reshape(tst_x, (tst_x.shape[0], 1, tst_x.shape[1]))

    # create and fit the LSTM network
    model_pred = Sequential()
    model_pred.add(LSTM(4, input_shape=(1, look_back)
                        , recurrent_regularizer='l1'
                        , activity_regularizer='l1'
                        , kernel_regularizer='l1'))
    model_pred.add(Dense(1))
    model_pred.compile(loss='mean_squared_error', optimizer='adam')
    model_pred.fit(tr_x, tr_y, epochs=1000, batch_size=1, verbose=0)

    # make predictions
    tr_predict = model_pred.predict(tr_x)
    if do_test:
        tst_predict = model_pred.predict(tst_x)
    prd1_x = ds2[len(ds2) - look_back - 1:len(ds2) - 1]
    prd1_x = numpy.array([numpy.array([prd1_x.flatten()])])
    prd_predict1 = model_pred.predict(prd1_x)
    prd2_x = ds2[len(ds2) - look_back:len(ds2)]
    prd2_x = numpy.array([numpy.array([prd2_x.flatten()])])
    prd_predict2 = model_pred.predict(prd2_x)
    prd3_x = numpy.append(prd2_x, prd_predict2)
    prd3_x = prd3_x[len(prd3_x) - look_back:len(prd3_x)]
    prd3_x = numpy.array([numpy.array([prd3_x.flatten()])])
    prd_predict3 = model_pred.predict(prd3_x)
    prd4_x = numpy.append(prd3_x, prd_predict3)
    prd4_x = prd4_x[len(prd4_x) - look_back:len(prd4_x)]
    prd4_x = numpy.array([numpy.array([prd4_x.flatten()])])
    prd_predict4 = model_pred.predict(prd4_x)
    prd5_x = numpy.append(prd4_x, prd_predict4)
    prd5_x = prd5_x[len(prd5_x) - look_back:len(prd5_x)]
    prd5_x = numpy.array([numpy.array([prd5_x.flatten()])])
    prd_predict5 = model_pred.predict(prd5_x)
    prd6_x = numpy.append(prd5_x, prd_predict5)
    prd6_x = prd6_x[len(prd6_x) - look_back:len(prd6_x)]
    prd6_x = numpy.array([numpy.array([prd6_x.flatten()])])
    prd_predict6 = model_pred.predict(prd6_x)
    prd7_x = numpy.append(prd6_x, prd_predict6)
    prd7_x = prd7_x[len(prd7_x) - look_back:len(prd7_x)]
    prd7_x = numpy.array([numpy.array([prd7_x.flatten()])])
    prd_predict7 = model_pred.predict(prd7_x)

    # invert predictions from scaled
    tr_predict = scaler.inverse_transform(tr_predict)
    tr_y = scaler.inverse_transform([tr_y])
    if do_test:
        tst_predict = scaler.inverse_transform(tst_predict)
        tst_y = scaler.inverse_transform([tst_y])
    prd_predict1 = scaler.inverse_transform(prd_predict1)
    print('Latest Prediction (in 1 days): %.0f' % prd_predict1)
    prd_predict2 = scaler.inverse_transform(prd_predict2)
    print('Latest Prediction (in 2 days): %.0f' % prd_predict2)
    prd_predict3 = scaler.inverse_transform(prd_predict3)
    print('Latest Prediction (in 3 days): %.0f' % prd_predict3)
    prd_predict4 = scaler.inverse_transform(prd_predict4)
    print('Latest Prediction (in 4 days): %.0f' % prd_predict4)
    prd_predict5 = scaler.inverse_transform(prd_predict5)
    print('Latest Prediction (in 5 days): %.0f' % prd_predict5)
    prd_predict6 = scaler.inverse_transform(prd_predict6)
    print('Latest Prediction (in 6 days): %.0f' % prd_predict6)
    prd_predict7 = scaler.inverse_transform(prd_predict7)
    print('Latest Prediction (in 7 days): %.0f' % prd_predict7)
    print(for_lstm.tail(40))

    # calculate root mean squared error
    tr_score = math.sqrt(mean_squared_error(tr_y[0], tr_predict[:, 0])) / tr_y.mean()
    print('Train Score: %.2f Scaled RMSE' % tr_score)
    if do_test:
        tst_score = math.sqrt(mean_squared_error(tst_y[0], tst_predict[:, 0])) / tst_y.mean()
        print('Test Score: %.2f Scaled RMSE' % tst_score)

    # Create a forecast series of 7 days
    forecast_7 = [prd_predict1[0, 0]
        , prd_predict2[0, 0]
        , prd_predict3[0, 0]
        , prd_predict4[0, 0]
        , prd_predict5[0, 0]
        , prd_predict6[0, 0]
        , prd_predict7[0, 0]]
    dates = pd.date_range(start=for_lstm.index.max() + pd.DateOffset(), periods=7)

    # shift train predictions for plotting
    tr_predict_plot = numpy.empty_like(ds2)
    tr_predict_plot[:, :] = numpy.nan
    tr_predict_plot[look_back:len(tr_predict) + look_back, :] = tr_predict

    # shift test predictions for plotting
    if do_test:
        tst_predict_plot = numpy.empty_like(ds2)
        tst_predict_plot[:, :] = numpy.nan
        tst_predict_plot[len(tr_predict) + (look_back * 2) + 1:len(ds2) - 1, :] = tst_predict

    # plot baseline and predictions
    plt.plot(numpy.array(for_lstm.index), scaler.inverse_transform(ds2))
    plt.plot(numpy.array(for_lstm.index), tr_predict_plot)
    plt.plot(numpy.array(dates), forecast_7)
    if do_test:
        plt.plot(numpy.array(for_lstm.index), tst_predict_plot)
    if perspective == 'vic':
        p_title = 'Victorian'
    elif perspective == 'nsw':
        p_title = 'New South Wales'
    elif perspective == 'qld':
        p_title = 'Queensland'
    elif perspective == 'global':
        p_title = 'Global'
    else:
        p_title = ''
    plt.title(p_title + ' Daily Covid19 Infections')
    plt.xlabel('Date')
    plt.ylabel('Count of Cases')
    if do_test:
        plt.legend(['Actual', 'Training', 'Forecast', 'Test'])
    else:
        plt.legend(['Actual', 'Training', 'Forecast'])

    return plt
