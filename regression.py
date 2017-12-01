import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error
import math
from sklearn import metrics

flights = pd.read_csv('data/discretized_flights_data.csv')

X = flights[['DAY_OF_YEAR','DAY_OF_WEEK','AIRLINE','FLIGHT_NUMBER','ORIGIN_AIRPORT',
						'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL', 
						'ARRIVAL_TIME','ELAPSED_TIME']]
y = flights['ARRIVAL_DELAY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

regression_model = linear_model.LinearRegression()
regression_model.fit(X_train, y_train)

y_predict_lr = regression_model.predict(X_test)
print 'Linear Regression Mean squared error: {}'.format(math.sqrt(mean_squared_error(y_predict_lr, y_test)))
print 'Linear Regression Median absolute error: {}'.format(median_absolute_error(y_predict_lr, y_test))

lasso = linear_model.Lasso()
lasso.fit(X_train, y_train)

y_predict_lasso = lasso.predict(X_test)
print 'Lasso Mean squared error: {}'.format(math.sqrt(mean_squared_error(y_predict_lasso, y_test)))
print 'Lasso Median absolute error: {}'.format(median_absolute_error(y_predict_lasso, y_test))

nn = MLPRegressor(verbose=True)
nn.fit(X_train, y_train)

y_predict_nn = nn.predict(X_test)
print 'Neural net Mean squared error: {}'.format(math.sqrt(mean_squared_error(y_predict_nn, y_test)))
print 'Neural net Median absolute error: {}'.format(median_absolute_error(y_predict_nn, y_test))