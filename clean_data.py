import pandas as pd
import datetime

#_________________________________________________________
# Function that convert the 'HHMM' string to datetime.time
def format_heure(chaine):
    if pd.isnull(chaine):
        return np.nan
    else:
        if chaine == 2400: chaine = 0
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure
#_____________________________________________________________________
# Function that combines a date and time to produce a datetime.datetime
def combine_date_heure(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])
#_______________________________________________________________________________
# Function that combine two columns of the dataframe to create a datetime format
def create_flight_time(df, col):    
    liste = []
    for index, cols in df[['DATE', col]].iterrows():    
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0,0)
            liste.append(combine_date_heure(cols))
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pd.Series(liste)


def clean_data():
	flights = pd.read_csv('data/flights.csv')
	flights.dropna(subset=['ARRIVAL_DELAY', 'ELAPSED_TIME'], inplace=True)
	
	flights['DAY_OF_YEAR'] = flights.apply(lambda row: datetime.date(2015, row.MONTH, row.DAY).timetuple().tm_yday, axis=1)

	variables_to_remove = ['TAIL_NUMBER','DEPARTURE_DELAY','TAXI_OUT', 'WHEELS_OFF',  
							'AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY',
							'WEATHER_DELAY', 'DISTANCE','WHEELS_ON','TAXI_IN','DIVERTED','CANCELLATION_REASON',
							'DEPARTURE_TIME','SCHEDULED_TIME', 'AIR_TIME', 'DAY_OF_WEEK']
	flights.drop(variables_to_remove, axis = 1, inplace = True)

	mask = (flights['ORIGIN_AIRPORT'].str.len() == 3) & (flights['DESTINATION_AIRPORT'].str.len() == 3)
	flights = flights.loc[mask]

	print 'about to get datetimes'

	# get datetimes 
	flights['DATE'] = pd.to_datetime(flights[['YEAR','MONTH', 'DAY']])
	flights['SCHEDULED_DEPARTURE'] = create_flight_time(flights, 'SCHEDULED_DEPARTURE')
	flights['SCHEDULED_ARRIVAL'] = create_flight_time(flights, 'SCHEDULED_ARRIVAL')
	print 'almost there!!'
	flights['ARRIVAL_TIME'] = create_flight_time(flights, 'ARRIVAL_TIME')

	flights.drop(['YEAR','MONTH','DAY', 'DATE'], axis=1, inplace=True)


# 	kept_vars = ['DAY_OF_WEEK','AIRLINE', 'FLIGHT_NUMBER','ORIGIN_AIRPORT',
#				'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','ELAPSED_TIME',
#				'SCHEDULED_ARRIVAL','ARRIVAL_TIME','ARRIVAL_DELAY','CANCELLED',]

	flights = flights[['DAY_OF_YEAR','AIRLINE','FLIGHT_NUMBER','ORIGIN_AIRPORT',
						'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL', 
						'ARRIVAL_TIME','ARRIVAL_DELAY','ELAPSED_TIME', 'CANCELLED']]

	flights.to_csv('data/official_clean_flights.csv', index=False)

clean_data()
