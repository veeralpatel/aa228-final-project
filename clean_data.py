import pandas as pd
import datetime
import pickle

def clean_data():
	flights = pd.read_csv('data/flights.csv')
	flights.dropna(subset=['ARRIVAL_DELAY', 'ELAPSED_TIME'], inplace=True)
	
	flights['DAY_OF_YEAR'] = flights.apply(lambda row: datetime.date(2015, row.MONTH, row.DAY).timetuple().tm_yday, axis=1)

	variables_to_remove = ['YEAR','MONTH','DAY','TAIL_NUMBER','DEPARTURE_DELAY','TAXI_OUT', 'WHEELS_OFF',  
							'AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY',
							'WEATHER_DELAY', 'DISTANCE','WHEELS_ON','TAXI_IN','DIVERTED','CANCELLATION_REASON',
							'DEPARTURE_TIME','SCHEDULED_TIME', 'AIR_TIME']
	flights.drop(variables_to_remove, axis = 1, inplace = True)

	mask = (flights['ORIGIN_AIRPORT'].str.len() == 3) & (flights['DESTINATION_AIRPORT'].str.len() == 3)
	flights = flights.loc[mask]

	airport_score = {}
	with open('airport_score.pickle', 'rb') as handle:
		airport_score = pickle.load(handle)
	flights['ORIGIN_SCORE'] = flights.apply(lambda row: airport_score[row.ORIGIN_AIRPORT], axis=1)
	flights['DESTINATION_SCORE'] = flights.apply(lambda row: airport_score[row.DESTINATION_AIRPORT], axis=1)

# 	kept_vars = ['DAY_OF_WEEK','AIRLINE', 'FLIGHT_NUMBER','ORIGIN_AIRPORT',
#				'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','ELAPSED_TIME',
#				'SCHEDULED_ARRIVAL','ARRIVAL_TIME','ARRIVAL_DELAY','CANCELLED',]

	flights = flights[['DAY_OF_YEAR','DAY_OF_WEEK','AIRLINE','FLIGHT_NUMBER','ORIGIN_AIRPORT',
						'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL', 
						'ARRIVAL_TIME','ARRIVAL_DELAY','ELAPSED_TIME','ORIGIN_SCORE','DESTINATION_SCORE']]

	flights.to_csv('data/clean_flights.csv', index=False)

clean_data()
