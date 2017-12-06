import pandas as pd
from collections import defaultdict
import pickle

flights_df = pd.read_csv('data/clean_flights.csv')
tot_flight_times = defaultdict(int)
tot_flight_counts = defaultdict(int)

for _, row in flights_df.iterrows():
    flight = row.AIRLINE + str(row.FLIGHT_NUMBER)
    tot_flight_times[(flight, row.ORIGIN_AIRPORT, row.DESTINATION_AIRPORT)] += row.ELAPSED_TIME
    tot_flight_counts[(flight, row.ORIGIN_AIRPORT, row.DESTINATION_AIRPORT)] += 1

# with open('tot_flight_times.pkl', 'wb') as f:
#     pickle.dump(tot_flight_times, f)
# with open('tot_flight_counts.pkl', 'wb') as f:
#     pickle.dump(tot_flight_counts, f)
print 'got flight times and counts'

avg_flight_times = defaultdict(int)
for key in tot_flight_times:
    avg_flight_times[key] = tot_flight_times[key] / tot_flight_counts[key]

print avg_flight_times
with open('avg_flight_times.pkl', 'wb') as f:
    pickle.dump(avg_flight_times, f)