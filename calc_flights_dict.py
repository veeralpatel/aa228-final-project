import functools, pickle
import pandas as pd
from collections import defaultdict

flights_df = pd.read_csv('data/official_clean_flights.csv')
# airport_to_flights = dict.fromkeys

dd_list = functools.partial(defaultdict, list)
airport_to_flights = defaultdict(dd_list)



for _, row in flights_df.iterrows():
    flight_num = row.AIRLINE + str(row.FLIGHT_NUMBER)
    flight = (flight_num, row.ORIGIN_AIRPORT, row.DESTINATION_AIRPORT, row.SCHEDULED_DEPARTURE, 
                row.ARRIVAL_TIME, row.ELAPSED_TIME)
    airport_to_flights[row.ORIGIN_AIRPORT][row.DAY_OF_YEAR].append(flight)

    if _ % 100000 == 0:  
        print 'on index ', _

with open('airport_to_flights_dict.pkl', 'wb') as f:
    pickle.dump(airport_to_flights, f)

# all flights from an origin
# Airport --> ((AA228, O, D, Departure_time, real_arrival_time, elapesed_time))

# dict --> dict --> set
# airport --> day_of_year --> ordered list of flights 