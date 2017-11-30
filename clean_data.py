import pandas as pd
import datetime

flights = pd.read_csv('data/flights.csv')
count_flights = flights['ORIGIN_AIRPORT'].value_counts()
flights['DAY_OF_YEAR'] = flights.apply(lambda row: datetime.date(2015, row.MONTH, row.DAY).timetuple().tm_yday, axis=1)

flights.to_csv('flights_with_DOY.csv', sep='\t', index=False)
