import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, median_absolute_error
import math
import matplotlib.pyplot as plt
from sklearn import metrics

flights = pd.read_csv('data/discretized_flights_data.csv')

X = flights[['DAY_OF_YEAR','DAY_OF_WEEK','AIRLINE','FLIGHT_NUMBER','ORIGIN_AIRPORT',
						'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL', 
						'ARRIVAL_TIME','ELAPSED_TIME']]
y = flights['ARRIVAL_DELAY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

regression_model = linear_model.LinearRegression()
regression_model.fit(X_train, y_train)

print('Coefficients: \n', regression_model.coef_)

y_predict = regression_model.predict(X_test)
print 'Mean squared error: {}'.format(math.sqrt(mean_squared_error(y_predict, y_test)))
print 'Median absolute error: {}'.format(median_absolute_error(y_predict, y_test))

