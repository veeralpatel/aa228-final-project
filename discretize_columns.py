import pandas as pd
from collections import defaultdict

def discretize_columns():
	flights = pd.read_csv('data/clean_flights.csv')
	
	all_airports = set(flights['ORIGIN_AIRPORT'].tolist() + flights['DESTINATION_AIRPORT'].tolist())
	all_airlines = set(flights['AIRLINE'].tolist())

	airport_dict = defaultdict(lambda: 0)
	airline_dict = defaultdict(lambda: 0)

	i = 1
	for airport in all_airports:
		if str(airport) not in airport_dict:
			airport_dict[str(airport)] = i
			i += 1

	j = 1
	for airline in all_airlines:
		if airline not in airline_dict:
			airline_dict[airline] = j
			j += 1

	flights['ORIGIN_AIRPORT'] = flights.apply(lambda row: airport_dict[row.ORIGIN_AIRPORT], axis=1)
	flights['DESTINATION_AIRPORT'] = flights.apply(lambda row: airport_dict[row.DESTINATION_AIRPORT], axis=1)
	flights['AIRLINE'] = flights.apply(lambda row: airline_dict[row.AIRLINE], axis=1)

	flights.to_csv('data/discretized_flights_data.csv', index=False)

discretize_columns()
