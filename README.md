# End2End_forecasting_solutions

#End2End Pipeline for timeseries forecating.

Input from user:

-Timeseries data(file path)
-Roll up choice (Want to aggregate data before forecasting. ex: daily data to monthly data)
-Date columns ('Date' by default)
-target variables and dependant variables(in case of multi variate)
-Exogenous Variables(if any)
-Forecast Horizon


The solution will detect the type of time series data, provide proprocessing, required Statistial analysis, model selection and model tuning. It will provide the forecast prediction.
