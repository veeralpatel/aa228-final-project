import pandas as pd
import pickle

def discretize_columns():
	flights = pd.read_csv('data/clean_flights.csv')
	
	all_airports = set(flights['ORIGIN_AIRPORT'].tolist() + flights['DESTINATION_AIRPORT'].tolist())
	all_airlines = set(flights['AIRLINE'].tolist())

	airport_dict = {}
	airline_dict = {}

	i = 0
	for airport in all_airports:
		if str(airport) not in airport_dict:
			airport_dict[str(airport)] = i
			i += 1

	j = 0
	for airline in all_airlines:
		if airline not in airline_dict:
			airline_dict[airline] = j
			j += 1

	flights['ORIGIN_AIRPORT'] = flights.apply(lambda row: airport_dict[row.ORIGIN_AIRPORT], axis=1)
	flights['DESTINATION_AIRPORT'] = flights.apply(lambda row: airport_dict[row.DESTINATION_AIRPORT], axis=1)
	flights['AIRLINE'] = flights.apply(lambda row: airline_dict[row.AIRLINE], axis=1)

	flights.to_csv('data/discretized_flights_data.csv', index=False)
	
	with open('airport_dict.pickle', 'wb') as handle:
	    pickle.dump(airport_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('airline_dict.pickle', 'wb') as handle2:
	    pickle.dump(airline_dict, handle2, protocol=pickle.HIGHEST_PROTOCOL)

discretize_columns()
