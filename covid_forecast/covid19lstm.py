# Required libraries
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


def prepare_data(dataset, look_back=1):
    """
    This function takes a data set and offsets x by look_back returning x and y arrays
    """

    dX, dY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dX.append(a)
        dY.append(dataset[i + look_back, 0])
    return numpy.array(dX), numpy.array(dY)


def get_data(perspective):
    """
    This function returns and formats data based on the data perspective chosen:
    'global' or 'vic'
    """

    # Read in data
    ssl._create_default_https_context = ssl._create_unverified_context

    if perspective == 'global':
        url = "https://pomber.github.io/covid19/timeseries.json"
        covid_data = pd.read_json('https://pomber.github.io/covid19/timeseries.json')
        masterdf = pd.DataFrame
        for i, row in covid_data.iterrows():
            df = pd.json_normalize(covid_data[covid_data.keys()[i]])
            df = df.assign(country=covid_data.keys()[i])
            if i == 0:
                masterdf = df
            else:
                masterdf = masterdf.append(df)

    if perspective == 'vic':
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
            cases = numpy.append(cases, nat_timeseries[i]["VIC"][0])
            if len(nat_timeseries[i]["VIC"]) > 1:
                deaths = numpy.append(deaths, nat_timeseries[i]["VIC"][1])
                recovered = numpy.append(recovered, nat_timeseries[i]["VIC"][2])
                tests = numpy.append(tests, nat_timeseries[i]["VIC"][3])
            else:
                deaths = numpy.append(deaths, 0)
                recovered = numpy.append(recovered, 0)
                tests = numpy.append(tests, 0)

        vic_timeseries = {"date": date, "cases": cases, "deaths": deaths, "recovered": recovered, "tests": tests}

        # Create data frame from json
        masterdf = pd.DataFrame.from_dict(vic_timeseries)
        masterdf.set_index(pd.to_datetime(vic_timeseries['date'], format='%Y-%m-%d'))

    # Calculate daily cases and sum to date
    masterdf['date'] = pd.to_datetime(masterdf['date'])
    if perspective == 'global':
        masterdf['daily_counts'] = masterdf['confirmed'].diff()
    if perspective == 'vic':
        masterdf['daily_counts'] = masterdf['cases'].diff()
    masterdf.loc[masterdf.daily_counts < 0, 'daily_counts'] = 0
    masterdf.loc[masterdf.daily_counts.isnull(), 'daily_counts'] = 0
    for_lstm = masterdf.groupby('date').agg('sum')
    if perspective == 'global':
        for_lstm = for_lstm.drop(['confirmed', 'deaths', 'recovered'], axis=1)
    if perspective == 'vic':
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
    trX, trY = prepare_data(tr, look_back)
    if do_test:
        tstX, tstY = prepare_data(tst, look_back)

    # reshape input to be [samples, time steps, features]
    trX = numpy.reshape(trX, (trX.shape[0], 1, trX.shape[1]))
    if do_test:
        tstX = numpy.reshape(tstX, (tstX.shape[0], 1, tstX.shape[1]))

    # create and fit the LSTM network
    modelPred = Sequential()
    modelPred.add(LSTM(4, input_shape=(1, look_back)
                       , recurrent_regularizer='l1'
                       , activity_regularizer='l1'
                       , kernel_regularizer='l1'))
    modelPred.add(Dense(1))
    modelPred.compile(loss='mean_squared_error', optimizer='adam')
    modelPred.fit(trX, trY, epochs=1000, batch_size=1, verbose=0)

    # make predictions
    trPredict = modelPred.predict(trX)
    if do_test:
        tstPredict = modelPred.predict(tstX)
    prd1X = ds2[len(ds2) - look_back - 1:len(ds2) - 1]
    prd1X = numpy.array([numpy.array([prd1X.flatten()])])
    prdPredict1 = modelPred.predict(prd1X)
    prd2X = ds2[len(ds2) - look_back:len(ds2)]
    prd2X = numpy.array([numpy.array([prd2X.flatten()])])
    prdPredict2 = modelPred.predict(prd2X)
    prd3X = numpy.append(prd2X, prdPredict2)
    prd3X = prd3X[len(prd3X) - look_back:len(prd3X)]
    prd3X = numpy.array([numpy.array([prd3X.flatten()])])
    prdPredict3 = modelPred.predict(prd3X)
    prd4X = numpy.append(prd3X, prdPredict3)
    prd4X = prd4X[len(prd4X) - look_back:len(prd4X)]
    prd4X = numpy.array([numpy.array([prd4X.flatten()])])
    prdPredict4 = modelPred.predict(prd4X)
    prd5X = numpy.append(prd4X, prdPredict4)
    prd5X = prd5X[len(prd5X) - look_back:len(prd5X)]
    prd5X = numpy.array([numpy.array([prd5X.flatten()])])
    prdPredict5 = modelPred.predict(prd5X)
    prd6X = numpy.append(prd5X, prdPredict5)
    prd6X = prd6X[len(prd6X) - look_back:len(prd6X)]
    prd6X = numpy.array([numpy.array([prd6X.flatten()])])
    prdPredict6 = modelPred.predict(prd6X)
    prd7X = numpy.append(prd6X, prdPredict6)
    prd7X = prd7X[len(prd7X) - look_back:len(prd7X)]
    prd7X = numpy.array([numpy.array([prd7X.flatten()])])
    prdPredict7 = modelPred.predict(prd7X)

    # invert predictions from scaled
    trPredict = scaler.inverse_transform(trPredict)
    trY = scaler.inverse_transform([trY])
    if do_test:
        tstPredict = scaler.inverse_transform(tstPredict)
        tstY = scaler.inverse_transform([tstY])
    prdPredict1 = scaler.inverse_transform(prdPredict1)
    print('Latest Prediction (in 1 days): %.0f' % prdPredict1)
    prdPredict2 = scaler.inverse_transform(prdPredict2)
    print('Latest Prediction (in 2 days): %.0f' % prdPredict2)
    prdPredict3 = scaler.inverse_transform(prdPredict3)
    print('Latest Prediction (in 3 days): %.0f' % prdPredict3)
    prdPredict4 = scaler.inverse_transform(prdPredict4)
    print('Latest Prediction (in 4 days): %.0f' % prdPredict4)
    prdPredict5 = scaler.inverse_transform(prdPredict5)
    print('Latest Prediction (in 5 days): %.0f' % prdPredict5)
    prdPredict6 = scaler.inverse_transform(prdPredict6)
    print('Latest Prediction (in 6 days): %.0f' % prdPredict6)
    prdPredict7 = scaler.inverse_transform(prdPredict7)
    print('Latest Prediction (in 7 days): %.0f' % prdPredict7)
    print(for_lstm.tail(40))

    # calculate root mean squared error
    trScore = math.sqrt(mean_squared_error(trY[0], trPredict[:, 0])) / trY.mean()
    print('Train Score: %.2f Scaled RMSE' % trScore)
    if do_test:
        tstScore = math.sqrt(mean_squared_error(tstY[0], tstPredict[:, 0])) / tstY.mean()
        print('Test Score: %.2f Scaled RMSE' % tstScore)

    # Create a forecast series of 7 days
    forecast_7 = [prdPredict1[0, 0]
        , prdPredict2[0, 0]
        , prdPredict3[0, 0]
        , prdPredict4[0, 0]
        , prdPredict5[0, 0]
        , prdPredict6[0, 0]
        , prdPredict7[0, 0]]
    dates = pd.date_range(start=for_lstm.index.max() + pd.DateOffset(), periods=7)

    # shift train predictions for plotting
    trPredictPlot = numpy.empty_like(ds2)
    trPredictPlot[:, :] = numpy.nan
    trPredictPlot[look_back:len(trPredict) + look_back, :] = trPredict

    # shift test predictions for plotting
    if do_test:
        tstPredictPlot = numpy.empty_like(ds2)
        tstPredictPlot[:, :] = numpy.nan
        tstPredictPlot[len(trPredict) + (look_back * 2) + 1:len(ds2) - 1, :] = tstPredict

    # plot baseline and predictions
    plt.plot(numpy.array(for_lstm.index), scaler.inverse_transform(ds2))
    plt.plot(numpy.array(for_lstm.index), trPredictPlot)
    plt.plot(numpy.array(dates), forecast_7)
    if do_test:
        plt.plot(numpy.array(for_lstm.index), tstPredictPlot)
    if perspective == 'vic':
        p_title = 'Victorian'
    elif perspective == 'global':
        p_title = 'Global'
    plt.title(p_title + ' Daily Covid19 Infections')
    plt.xlabel('Date')
    plt.ylabel('Count of Cases')
    if do_test:
        plt.legend(['Actual', 'Training Series', 'Forecast', 'Test Series'])
    else:
        plt.legend(['Actual', 'Training Series', 'Forecast'])

    return plt
