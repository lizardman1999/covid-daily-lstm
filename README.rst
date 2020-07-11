Forecast New Daily Covid-19 Cases
=================================

This project uses Keras LSTM to produce a feed forward neural net forecast of daily cases.

How to install and use
----------------------

1. Navigate to the root diretory
2. Run the following:

	.. code-block:: bash

		pip3 install -r requirements.txt
	
		pip3 install ./covid_forecast 
	
3. Open a python3 prompt and run the following:

	.. code-block:: python

	 	import covid_forecast as cf
	 
	 	cf.run_daily_stats(persepctive='global',train_sample_size = 0.8)
		

Input Parameters
----------------

The perspective parameter may be either 'global' for global daily statistics or 'vic' for daily satistics for the state of Victoria in Australia. The training sample is a value > 0 and less than or equal to one. If this value is set to 1 all observations are used to train the forecast. This is the percentage of observations held out for training. The most recent observations are  used for testing the fit. By default this program "looks back" 10 days in training the model and forecasts forward 7 days. These values will be parametised in future development.

