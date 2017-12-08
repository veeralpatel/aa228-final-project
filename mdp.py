import util, math, random, datetime, sys, argparse
from collections import defaultdict
from util import ValueIteration
import numpy as np
import cPickle as pickle
from scipy.stats import truncnorm
import numpy as np

np.random.seed(11)

stochastic_rewards = defaultdict(float)

def hours_between(d1, d2):
    return divmod((d1 - d2).total_seconds(), 3600)[0]

def minutes_between(d1, d2):
    return divmod((d1 - d2).total_seconds(), 60)[0]

def convertToDateTime(str):
    return datetime.datetime.strptime(str, '%Y-%m-%d %H:%M:%S')

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class FlightMDP(util.MDP):
    def __init__(self, initial_origin, final_destination, start_time, prune_direct):
        self.initial_origin = initial_origin
        self.start_time = start_time
        self.final_destination = final_destination
        self.prune_direct = prune_direct

    # Return the start state.
    # States are your current airport and datetime
    def startState(self):
        # current location, current time, legs taken
        return (self.initial_origin, self.start_time, 0)

    # Return set of actions possible from |state|.
    # Return all the flights you can take from an airport that are after your current time
    def actions(self, state):
        all_actions = []

        origin = state[0]
        currentTime = state[1]
        currentNumLegs = state[2]

        if origin == self.final_destination:
            return [('DONE',None,None,None,None)]
        if origin == 'QUIT':
            return [('DONE',None,None,None,None)]

        today_tomorrow_flights = all_flights[origin][state[1].timetuple().tm_yday] + all_flights[origin][state[1].timetuple().tm_yday+1]
        for flight in today_tomorrow_flights:
            flightNumber = flight[0]
            originAirport = flight[1]
            destinationAirport = flight[2]
            scheduledDeparture = convertToDateTime(flight[3])
            arrivalTime = convertToDateTime(flight[4])
            elapsedTime = flight[5]

            if (self.prune_direct):
                if state == self.startState() and origin == self.initial_origin and destinationAirport == self.final_destination:
                    continue;

            if hours_between(scheduledDeparture, self.start_time) > 24:
                break 

            if arrivalTime > scheduledDeparture and scheduledDeparture > currentTime:
                # flight number, departure time, destination, real arrival time, elapsed time
                all_actions.append((flightNumber, scheduledDeparture, destinationAirport, arrivalTime, elapsedTime)) 

        if len(all_actions) == 0 or (currentNumLegs == 3 and origin != self.final_destination):
            return [('QUIT',None,None,None,None)]
        else:
            return all_actions

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (i.e. reaching the final destination airport) by setting the current airport to self.final_destination
    # * and returning an empty list (no more actions to take)
    def succAndProbReward(self, state, action):
        if action[0] == 'DONE':
            return []
        elif action[0] == 'QUIT':
            return [(('QUIT',None,state[2]),1.0,-100000)]

        currentLocation = state[0]
        currentDatetime = state[1]
        currentNumLegs = state[2]

        flight_number = action[0]
        scheduled_departure = action[1]
        destination = action[2]
        real_arrival_time = action[3]
        elapsed_time = action[4]

        if currentNumLegs > 3:
            return [(('QUIT',None,state[2]),1.0,-100000)]

        if currentLocation == self.final_destination:
            return []
        else:
            delta_between_flights = -1*minutes_between(scheduled_departure, currentDatetime)

            # cancelled flight 
            succCancelled = (currentLocation, scheduled_departure, currentNumLegs)
            probCancelled = 0.2
            rewardCancelled = delta_between_flights
            cancelled_flight = (succCancelled, probCancelled, rewardCancelled)

            # regular flight, not cancelled
            succ = (destination, real_arrival_time, currentNumLegs + 1)
            prob = 0.8
            noisy_elapsed_time = 0
            if action in stochastic_rewards:
                noisy_elapsed_time = stochastic_rewards[action]
            else:
                stochastic_rewards[action] = get_truncated_normal(mean=elapsed_time, sd=30, low=elapsed_time, upp=elapsed_time+120).rvs()
                noisy_elapsed_time = stochastic_rewards[action]
            reward = delta_between_flights - noisy_elapsed_time
            good_flight = (succ, prob, reward)
            actions = [cancelled_flight, good_flight]

            return actions

    def discount(self):
        return 1


# parse input args
parser = argparse.ArgumentParser()
parser.add_argument('--pruneDirect', action='store', dest='prune_direct', type=bool, default=False)
parser.add_argument('--epsilon', action='store', dest='epsilon', type=float, default=0.1)
parser.add_argument('--origin', action='store', dest='initial_origin', type=str, default='EWR')
parser.add_argument('--destination', action='store', dest='final_destination', type=str, default='SFO')
parser.add_argument('--outputPolicyFN', action='store', dest='output_policy_fn', type=str, default='optimal_policy.pkl')
parser.add_argument('--outputValueFN', action='store', dest='output_value_fn', type=str, default='value_function.pkle')
results = parser.parse_args()

# set vars to parser arguments or defaults
prune_direct = results.prune_direct
epsilon = results.epsilon
initial_origin = results.initial_origin
destination = results.final_destination
policy_filename = results.output_policy_fn
value_filename = results.output_value_fn

print 'loading pkl'
all_flights = {}
with open(r"airport_to_flights_dict.pkl", "rb") as input_file:
    all_flights = pickle.load(input_file)
print 'done loading pkl'

np.random.seed(11)

mdp = FlightMDP(initial_origin=initial_origin, start_time=datetime.datetime(2015, 1, 11, 8, 30), final_destination=destination, prune_direct=prune_direct)
alg = util.ValueIteration()
alg.solve(mdp, epsilon)


with open(value_filename, 'wb') as handle:
    pickle.dump(alg.V, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(policy_filename, 'wb') as handle:
    pickle.dump(alg.pi, handle, protocol=pickle.HIGHEST_PROTOCOL)

print 'dumped new policies'
print 'printing final path'

state = mdp.startState()
path = [(state, None)]
while True:
    print '\n'
    print 's: ', state, alg.V[state]
    if state[0] == mdp.final_destination:
        break
    else:
        action = alg.pi[state]
        print 'a: ', action
        state = (action[2],action[3],state[2]+1)
        path.append((state,action,state[2]+1))

