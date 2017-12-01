import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error
import math
from sklearn.neural_network import MLPRegressor
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler  

flights = pd.read_csv('data/discretized_flights_data.csv')

X = flights[['DAY_OF_YEAR','DAY_OF_WEEK','AIRLINE','FLIGHT_NUMBER','ORIGIN_AIRPORT',
						'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL']]
y = flights['ARRIVAL_DELAY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)

nn = MLPRegressor(verbose=True, hidden_layer_sizes=(50,))
nn.fit(X_train, y_train)

y_predict_nn = nn.predict(X_test)
print 'Neural net Mean squared error: {}'.format(math.sqrt(mean_squared_error(y_predict_nn, y_test)))
print 'Neural net Median absolute error: {}'.format(median_absolute_error(y_predict_nn, y_test))

joblib.dump(nn, 'model.pkl')