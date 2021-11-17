# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:44:09 2021

@author: faraz.akhtar
"""



#################################################################################################################################

#The pipeline is the main function that calls all other function


def pipeline(data, forecast_horizon, freq, series_type):
  print('Ingesting Data....')

  horizon_dates = horizon(data, forecast_horizon, freq, series_type)

  print(f'Total datapoints: {data.shape[0]}')

  data.interpolate(method = 'time', inplace = True) #fill missing values


  print('Checking for missing Dates....')
  if missing_dates(data) > 0:
    print(f'{round(missing_dates(data), 2)}% of dates are missing from dataset!')
    print('Imputing Data....')
    #imputing the missing date and its values
    data_imputed = missing_values(data,columns)
    print('Imputation Done. New Dataframe formed...')
  else: 
    print('No missing date!!!')
    data_imputed = data.copy()
  
#roll up the data if possible. 
  if series_type != rollup_level:
    print('Checking Rollup...')
    data_imputed = if_roll_up_possible(data_imputed, series_type, rollup_level)


  #checking stationarity of data
  #return stationarity = True/False, columns which are not stationarity, parameter d, log_transform = True/False
  stationarity, non_stationary_columns, d, log_transform = stationarity_check(data_imputed, columns)

  #check for seasonality
  #return seasonality = True/False and order of seasonality(m)
  seasonality, seasonal_order = check_seasonality(data, columns)
  trend = False


  baseline_model_prediction = forecast_baseline(data_imputed[columns[0]], d[0], horizon_dates)
  print(pd.DataFrame({
      'Date': horizon_dates,
      columns[0] : baseline_model_prediction
      }))
  models = check_models(data_imputed, columns, trend, seasonality, exo, horizon_dates, d, seasonal_order, log_transform)
  predicted = models
  print(pd.DataFrame({
      'Date': horizon_dates,
      columns[0] : predicted
      }))
  return pd.DataFrame({
      'Date': horizon_dates,
      columns[0] : predicted
      })





##################################################################################################################################

#Function to calculated the dates for the forcast horizon given by the user
#Example forecast horizon = 4, means forecast for next 4 days after where data ends
#return the dates for the given horizon

def horizon(data, forecast_horizon, freq, series_type):  #Input: Data, number of forecast, frequency of date, frequency of series
  a =  max(data.index)
  if series_type == 0:
    b = timedelta(days = 1)
    #a = a + b
    dates = []
    x = 0
    while x < forecast_horizon:
      a = a + b
      dates.append(a.to_pydatetime())
      x += 1



  if series_type == 1:
    b = timedelta(weeks = 1)
    #a = a + b
    dates = []
    x = 0
    while x < forecast_horizon:
      a = a + b
      dates.append(a.to_pydatetime())
      x += 1


  if series_type == 2:
    b = timedelta(days = 30 )
    #a = a + b
    dates = []
    x = 0
    while x < forecast_horizon:
      a = a + b
      dates.append(a.to_pydatetime())
      x += 1


  if series_type == 3:
    b = timedelta(weeks = 13)
    #a = a + b
    dates = []
    x = 0
    while x < forecast_horizon:
      a = a + b
      dates.append(a.to_pydatetime())
      x += 1

  return dates




##############################################################################################

#Check for the missing values and dates

def missing_values(data,columns):
    df1 = pd.DataFrame(pd.date_range(data.index[0],data.index[-1]-timedelta(days=1),freq='d'), columns = [index])
    df = data.reset_index()[[index] + columns]
    df1 = df1.merge(df, on = index, how = 'left').copy()
    #new_col = columns[0] + '_imputed'
    
    df1.set_index(index, inplace = True, drop = True)
    for col in columns:
      df1 = interpolate_miss(df1, col)
    return df1



def interpolate_miss(data, col):
    #new_col = col + '_imputed'
    data[col] = data[col].interpolate(method = 'time')
    return data



def missing_dates(data):
    if (data.index[1] - data.index[0]).days == 1:
      should_be = (data.index[-1] - data.index[0]).days
      total = data.shape[0]
      return ((should_be - total)/total)*100

    elif (data.index[1] - data.index[0]).days == 7:
      should_be = ((data.index[-1] - data.index[0]).days) / 7
      total = data.shape[0]
      return ((should_be - total)/total)*100

    elif (data.index[1] - data.index[0]).days in [28, 29, 30 ,31] :
      should_be = ((data.index[-1] - data.index[0]).days) / 30
      total = data.shape[0]
      return ((should_be - total)/total)*100

    elif (data.index[1] - data.index[0]).days > 360:
      should_be = ((data.index[-1] - data.index[0]).days) / 365
      total = data.shape[0]
      return ((should_be - total)/total)*100



#############################################################################################


#check for rollup and do the rollup if possible

def if_roll_up_possible(data, series, rollup_level):
    if series > rollup_level:
        a = series
        print(f'Roll up not possible. Please enter from these: {level[a:]}')
    else:
        dg = data.groupby(pd.Grouper(freq= frequency[rollup_level])).sum()
        if dg.shape[0] < 50:
            print('Data not sufficient')
        else:
          print(f'Rollup done. Total datapoints after Rollup: {dg.shape[0]}')
          return dg
        


#############################################################################################


def d_value(series):
  i = 1
  adi = 1
  while (adi > 0.05) & (i < 5):
    series = series.diff().dropna()
    adi = adfuller(series)[1]
    i += 1
  return i
#############################################################################################

# check for stationarity of data
#return stationarity = True/False, columns which are not stationarity, parameter d, log_transform = True/False

def stationarity_check(data, columns):
  print('Checking Stationarity of data...')
  dvalue = []
  non_stationary_columns = []
  stationary = True
  for col in columns:
    log_transform = False
    s = data[col]
    ad = adfuller(s)
    if ad[1] < 0.05:
      print(f'{col} data is stationary')
      dvalue.append(0)
    else:
      non_stationary_columns.append(col)
      print(f'{col} data is not stationary...')
      print('checking for d value...')
      ds = d_value(s)
      if ds > 2:
        print('Transforming into log data')
        s = log(s)
        log_transform = True
        ds = d_value(s)
      dvalue.append(ds)

      stationary = False
  if stationary == True:
    print('Overall data is Stationary')
  else:
    print('Overall data is not Stationary')
  return stationary, non_stationary_columns, dvalue, log_transform


#############################################################################################

#check for the seasonality in data

def check_seasonality(data, columns):
  s = data[columns[0]]
  decomposition = seasonal_decompose(s, extrapolate_trend = 'freq',  model = 'additive')
  seasonal = decomposition.seasonal
  seasonal = seasonal.to_frame()
  mean = seasonal.iloc[2,0]
  std = seasonal.iloc[3, 0]
  done = False
  check = 0
  i = 1
  while done == False:
    if (seasonal.iloc[i, 0] - 10) <= seasonal.iloc[0,0] <= (seasonal.iloc[i, 0] + 10):
      check = 0
      for j in range(10):
        if (seasonal.iloc[i + j, 0] - 10) <= seasonal.iloc[j,0] <= (seasonal.iloc[i + j, 0] + 10):
          check += 1
          if check >= 5:
            done = True
            seasonality = True
          print (i)
    i += 1

  else:
    seasonality = False
  #check Seasonality
  return seasonality, i


##############################################################################################


def forecast_baseline(series, d, horizon):
  X = series.values
  X = [float(x) for x in X]
  size = int(len(X) * 0.7)
  train, test = X[0:size], X[size:len(X)]
  history = [x for x in train]
  predictions = list()
  # walk-forward validation
  for t in range(len(test)):
	  model = ARIMA(history, order=(0,d,0))
	  model_fit = model.fit()
	  output = model_fit.forecast()
	  yhat = output[0]
	  predictions.append(yhat)
	  obs = test[t]
	  history.append(obs)
	  #print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
  rmse = sqrt(mean_squared_error(test, predictions))
  print('Test RMSE: %.3f' % rmse)
  #plot forecasts against actual outcomes
  plt.plot(test)
  plt.plot(predictions, color='red')
  plt.show()


  X = series.values
  X = [float(x) for x in X]
  #size = int(len(X) * 0.7)
  #train, test = X[0:size], X[size:len(X)]
  history = [x for x in X]
  predictions = list()
  # walk-forward validation
  for t in range(len(horizon)):
	  model = ARIMA(history, order=(0,d,0))
	  model_fit = model.fit()
	  output = model_fit.forecast()
	  yhat = output[0]
	  predictions.append(yhat)
	  #obs = test[t]
	  history.append(yhat)
  return [x[0] for x in predictions] 


###############################################################################################


def check_models(data, columns, trend, seasonality, exo, horizon_dates, d_value, seasonal_order, log_transform):
  if len(columns) == 1:                       #univariate
    if trend == False:
      if seasonality == False:
        if  exo == None:
          models = forecast_arima(data[columns[0]], d_value[0], horizon_dates, log_transform)
        else:
          models = forecast_arimax(data[columns[0]], exo, d_value[0], horizon_dates, log_transform)
      else:
        if exo == None:
          models = forecast_Sarima(data[columns[0]], d_value[0], horizon_dates, seasonal_order,log_transform)
        else:
          models = forecast_Sarimax(data[columns[0]], exo, d_value[0], horizon_dates, log_transform)
    else:
      if seasonality == False:
        if  exo == None:
          models = forecast_arima(data[columns[0]], d_value[0], horizon_dates, log_transform)
        else:
          models = forecast_arimax(data[columns[0]], exo, d_value[0], horizon_dates, 'log_transform')
      else:
        if exo == None:
          models = forecast_Sarima(data[columns[0]], d_value[0], horizon_dates, seasonal_order,log_transform)
        else:
          models = forecast_Sarimax(data[columns[0]], exo, d_value[0], horizon_dates)
  else:
    if trend == False:
      if seasonality == False:
        if  exo == None:
          models = lstm(data, 4)
        else:
          models = lstm(data, 4)
      else:
        if exo == None:
          models = lstm(data, 4)
        else:
          models = lstm(data, 4)
    else:
      if seasonality == False:
        if  exo == None:
          models = lstm(data, 4)
        else:
          models = lstm(data, 4)
      else:
        if exo == None:
          models = lstm(data, 4)
        else:
          models = lstm(data, 4)


##############################################################################################

def forecast_arima(series, d, horizon, log_transform):
	original_series = series.copy()
	series_diff = series.copy()
	diff = 0

	if log_transform:
		series_diff = log(series_diff)
	
	print(f'D: {d}')
	while diff < d:
		series_diff = series_diff.diff().dropna()
		diff += 1
	acf_values = acf(series_diff.values)
	pacf_values = pacf(series_diff.values)
	n = series_diff.shape[0]
	crit_value = (exp(2*1.96/sqrt(n - 3)-1))/(exp(2*1.96/sqrt(n-3)+1))
	qs = []
	ps = []
	i = 1
	while i < 5:
		if (acf_values[1] > crit_value) or (acf_values[1] < -crit_value):
			qs.append(i)
		i += 1
	for j in range(1, len(qs)):
		if (pacf_values[j] > crit_value) or (acf_values[1] < -crit_value):
			ps.append(j)
	print(qs, ps)
	rmse = float('inf')
	order_best = (0, d, 0)



	if log_transform:
		series = log(series)


	for q in qs:
		for p in ps:
			order_temp = (p, d, q)
			print(order_temp)
			#val = series.values
			X = series.values
			size = int(len(X) * 0.7)
			train, test = X[0:size], X[size:len(X)]
			history = [x for x in train]
			predictions = list()
			try:
				for t in range(len(test)):
					model = ARIMA(history, order=order_temp)
					model_fit = model.fit(transparams = True)
					output = model_fit.forecast()
					yhat = output[0]
					predictions.append(yhat)
					obs = test[t]
					history.append(obs)
	  		#print('predicted=%f, expected=%f' % (yhat, obs))
		# evaluate forecasts
				rmse_temp = sqrt(mean_squared_error(test, predictions))
				print(rmse_temp)
				if rmse_temp < rmse:
					rmse = rmse_temp
					order_best = order_temp
					best = predictions

			except:
				pass


	if rmse == float('inf'):
		print('Baseline model shows best forecast')
		best = forecast_baseline(original_series, d, horizon)
	
	else:
		val = series.values
		history = [x for x in val]
		predictions = list()
		for t in range(len(horizon)):
			model = ARIMA(history, order=order_best)
			model_fit = model.fit(transparams = False)
			output = model_fit.forecast()
			yhat = output[0]
			predictions.append(yhat)
			#obs = test[t]
			history.append(yhat)
	  	#print('predicted=%f, expected=%f' % (yhat, obs))


			print(f'Total RMSE: {rmse}')
			plt.plot(test)
			plt.plot(best, color='red')
			plt.show()
			print(f'best order for ARIMA model: {order_best}')
 

	if log_transform:
		forecasted_data = [exp(x[0]) for x in predictions]

	else:
		forecasted_data = [x[0] for x in predictions]


	return forecasted_data



##############################################################################################


def forecast_arimax(series, exo, d, horizon, log_transform):
	original_series = series.copy()
	series_diff = series.copy()
	diff = 0

	if log_transform:
		series_diff = log(series_diff)
	
	print(f'D: {d}')
	while diff < d:
		series_diff = series_diff.diff().dropna()
		diff += 1
	acf_values = acf(series_diff.values)
	pacf_values = pacf(series_diff.values)
	n = series_diff.shape[0]
	crit_value = (exp(2*1.96/sqrt(n - 3)-1))/(exp(2*1.96/sqrt(n-3)+1))
	qs = []
	ps = []
	i = 1
	while i < 5:
		if (acf_values[1] > crit_value) or (acf_values[1] < -crit_value):
			qs.append(i)
		i += 1
	for j in range(1, len(qs)):
		if (pacf_values[j] > crit_value) or (acf_values[1] < -crit_value):
			ps.append(j)
	print(qs, ps)
	rmse = float('inf')
	order_best = (0, d, 0)



	if log_transform:
		series = log(series)


	for q in qs:
		for p in ps:
			order_temp = (p, d, q)
			print(order_temp)
			#val = series.values
			X = series.values
			size = int(len(X) * 0.7)
			train, test = X[0:size], X[size:len(X)]
			history = [x for x in train]
			predictions = list()
			try:
				for t in range(len(test)):
          #model = sm.tsa.statespace.SARIMAX(history, order - order_temp, seasonal_order = (0,0,0,0) exog = exo)
					model = sm.tsa.statespace.SARIMAX(history, order - order_temp, seasonal_order = (0,0,0,0), exog = exo)
					model_fit = model.fit(transparams = True)
					output = model_fit.forecast()
					yhat = output[0]
					predictions.append(yhat)
					obs = test[t]
					history.append(obs)
	  		#print('predicted=%f, expected=%f' % (yhat, obs))
		# evaluate forecasts
				rmse_temp = sqrt(mean_squared_error(test, predictions))
				print(rmse_temp)
				if rmse_temp < rmse:
					rmse = rmse_temp
					order_best = order_temp
					best = predictions

			except:
				pass


	if rmse == float('inf'):
		print('Baseline model shows best forecast')
		best = forecast_baseline(original_series, d, horizon)
	
	else:
		val = series.values
		history = [x for x in val]
		predictions = list()
		for t in range(len(horizon)):
			model = sm.tsa.statespace.SARIMAX(history, order - order_temp, seasonal_order = (0,0,0,0), exog = exo)
			model_fit = model.fit(transparams = False)
			output = model_fit.forecast()
			yhat = output[0]
			predictions.append(yhat)
			#obs = test[t]
			history.append(yhat)
	  	#print('predicted=%f, expected=%f' % (yhat, obs))


			print(f'Total RMSE: {rmse}')
			plt.plot(test)
			plt.plot(best, color='red')
			plt.show()
			print(f'best order for ARIMA model: {order_best}')
 

	if log_transform:
		forecasted_data = [exp(x[0]) for x in predictions]

	else:
		forecasted_data = [x[0] for x in predictions]


	return forecasted_data


###############################################################################################


def forecast_Sarima(series, d, horizon, m,log_transform):
  original_series = series.copy()
  series_diff = series.copy()
  diff = 0

  if log_transform:
	  series_diff = log(series_diff)
	
  print(f'D: {d}')
  while diff < d:
    series_diff = series_diff.diff().dropna()
    diff += 1
  acf_values = acf(series_diff.values)
  pacf_values = pacf(series_diff.values)
  n = series_diff.shape[0]
  crit_value = (exp(2*1.96/sqrt(n - 3)-1))/(exp(2*1.96/sqrt(n-3)+1))
  qs = []
  ps = []
  i = 1
  while i < 5:
    if (acf_values[1] > crit_value) or (acf_values[1] < -crit_value):
      qs.append(i)
    i += 1
  for j in range(1, len(qs)):
    if (pacf_values[j] > crit_value) or (acf_values[1] < -crit_value):
      ps.append(j)
  print(qs, ps)
  rmse = float('inf')
  order_best = (0, d, 0)



  if log_transform:
	  series = log(series)


  for q in qs:
    for p in ps:
      order_temp = (p, d, q)
      seasonal_order_temp = (p, d, q, m)
      print(order_temp)
			#val = series.values
      X = series.values
      size = int(len(X) * 0.7)
      train, test = X[0:size], X[size:len(X)]
      history = [x for x in train]
      predictions = list()
      try:
        for t in range(len(test)):
          #model = sm.tsa.statespace.SARIMAX(history, order - order_temp, seasonal_order = (0,0,0,0) exog = exo)
          model = sm.tsa.statespace.SARIMAX(history, order - order_temp, seasonal_order = seasonal_order_temp, exog = exo)
          model_fit = model.fit(transparams = True)
          output = model_fit.forecast()
          yhat = output[0]
          predictions.append(yhat)
          obs = test[t]
          history.append(obs)
		# evaluate forecasts
        rmse_temp = sqrt(mean_squared_error(test, predictions))
        print(rmse_temp)
        if rmse_temp < rmse:
          rmse = rmse_temp
          order_best = order_temp
          best = predictions

      except:
        pass


  if rmse == float('inf'):
    print('Baseline model shows best forecast')
    best = forecast_baseline(original_series, d, horizon)
	
  else:
    val = series.values
    history = [x for x in val]
    predictions = list()
    for t in range(len(horizon)):
      model = sm.tsa.statespace.SARIMAX(history, order - order_temp, seasonal_order = (0,0,0,0), exog = exo)
      model_fit = model.fit(transparams = False)
      output = model_fit.forecast()
      yhat = output[0]
      predictions.append(yhat)
      history.append(yhat)
	  	

      print(f'Total RMSE: {rmse}')
      plt.plot(test)
      plt.plot(best, color='red')
      plt.show()
      print(f'best order for ARIMA model: {order_best}')
 

  if log_transform:
    forecasted_data = [exp(x[0]) for x in predictions]

  else:
    forecasted_data = [x[0] for x in predictions]


  return forecasted_data

#############################################################################################


def lstm(data, forecast_horizon):
  n_rows = data.shape[0]
  N_features = data.shape[1]
  b = int(n_rows / 5)
  n = int(n_rows / 10)
  scaler = MinMaxScaler()
  data_scaled = scaler.fit_transform(data)
  features = data_scaled
  target = data_scaled[:,0]
  #TimeseriesGenerator(features, target, length = n, sampling_rate = 1 , batch_size = b)[]
  x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, shuffle = False)
  train_generator = TimeseriesGenerator(x_train, y_train, length = n, sampling_rate = 1 , batch_size = b)
  test_generator = TimeseriesGenerator(x_test, y_test, length = n, sampling_rate = 1 , batch_size = b)


  model = tf.keras.Sequential()
  model.add(tf.keras.layers.LSTM(128, input_shape = (n, N_features), return_sequences = True))
  model.add(tf.keras.layers.LeakyReLU(alpha = 0.5))
  model.add(tf.keras.layers.LSTM(128, return_sequences = True))
  model.add(tf.keras.layers.LeakyReLU(alpha = 0.5))
  model.add(tf.keras.layers.Dropout(0.3))
  model.add(tf.keras.layers.LSTM(64, return_sequences = False))
  model.add(tf.keras.layers.Dropout(0.3))
  model.add(tf.keras.layers.Dense(1))

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min')


  model.compile(loss = tf.losses.MeanAbsoluteError(),
              optimizer = tf.optimizers.Adam(),
              metrics = [tf.metrics.MeanAbsoluteError()])


  history = model.fit_generator(train_generator, epochs= 100, validation_data= test_generator,
                              shuffle = False, callbacks = [early_stopping])
  
  predictions = model.predict_generator(test_generator)
  df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(y_test[n:])], axis = 1)
  #rev = scaler.inverse_transform(df_pred)
  columns =  ['predicted', 'actual']
  df_pred.columns = columns
  #df_pred['predicted'] = rev[:,0]
  #df_pred['actual'] = rev[:,1]

  df_pred.plot()

  return df_pred
