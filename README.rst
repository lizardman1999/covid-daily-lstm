Forecast New Daily Covid-19 Cases
=================================

This project uses Keras LSTM to produce a feed forward neural net forecats of daily cases.

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

The perspective parameter may be either 'global' for global daily statistics or 'vic' for daily satistics for the state of Victoria in Australia. The training sample is a value > 0 and less than or eual to one. Thsi is the percentage of observations held out for training. This is an out of time sample. The remaining values will be used for testing the fit. By default this program "lokks back" 10 days and forecasts forward 7 days. This will be parametised in future iterations.

