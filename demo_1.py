import fileinput
import os
import warnings
warnings.filterwarnings('ignore')
#from pylab import rcParams
#rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api
#from statsmodels.tsa.Sarima_model import SARIMA
#from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import pandas as pd
#from nsepy import get_history
from datetime import date, timedelta
from datetime import datetime
import matplotlib.pyplot as plt
#import seaborn as sns
from statsmodels.tsa.stattools import adfuller
#import holidays
#%matplotlib inline
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.simplefilter("ignore")
from math import sqrt
from math import exp
from numpy import log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import cgi, os
import cgitb; cgitb.enable()
form = cgi.FieldStorage()
import tensorflow as tf 



#Import all Functions from function.py
from function import *


#Defining Input dataType


#Defining Input Choices

dimensions = ['Univariate', 'Multivariate']      #Dimension of data
levels = ['daily', 'weekly', 'monthly', 'quartely', 'yearly']     #time level of data
columns = []                            #Target Features followed by dependent variables

rollup_aggregations = ['Count', 'Average' ]
frequency = ['1D', '7D', '1M', '3M', '12M']
forecast_horizon = None                 #number of forecast data (integer)
index = 'Date'                          #By default Data column will be used as index





#Input given by Users
Data = None                  #Data Provided by user
dimension = None             #chosen from dimension list
series_type = None           #index of level chosen
rollup_level = None          #index of level chosen
rollup_aggregation = None    #chosen from rollup_aggregations
columns = None               # to be specified by users (will be part of columns list)
index = None                 #index column other than DATE
forecast_horizon = None
freq = None
exo = None                   #exogenous variables(in form of lists of Dataframe)



Pipeline(Data)



