import pandas as pd
from collections import defaultdict
import pickle

flights_df = pd.read_csv('data/clean_flights.csv')
tot_flight_cancellations = defaultdict(int)
tot_flight_counts = defaultdict(int)

for _, row in flights_df.iterrows():
    flight = row.AIRLINE + str(row.FLIGHT_NUMBER)
    tot_flight_cancellations[(flight, row.ORIGIN_AIRPORT, row.DESTINATION_AIRPORT)] += row.CANCELLED
    tot_flight_counts[(flight, row.ORIGIN_AIRPORT, row.DESTINATION_AIRPORT)] += 1

# with open('tot_flight_times.pkl', 'wb') as f:
#     pickle.dump(tot_flight_times, f)
# with open('tot_flight_counts.pkl', 'wb') as f:
#     pickle.dump(tot_flight_counts, f)
print 'got flight times and counts'

chance_of_cancellation = defaultdict(int)
for key in tot_flight_cancellations:
    chance_of_cancellation[key] = 1.0 * tot_flight_cancellations[key] / tot_flight_counts[key]

print chance_of_cancellation
with open('transition_probabilities.pkl', 'wb') as f:
    pickle.dump(chance_of_cancellation, f)