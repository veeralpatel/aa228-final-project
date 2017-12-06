import util, math, random
from collections import defaultdict
from util import ValueIteration
import datetime

all_flights = {}
with open(r"someobject.pickle", "rb") as input_file:
    all_flights = cPickle.load(input_file)

def hours_between(d1, d2):
    return divmod((d1 - d2).total_seconds(), 3600)[0]

class FlightMDP(util.MDP):
    def __init__(self, origin, final_destination, start_time):
        self.origin = origin
        self.start_time = start_time
        self.final_destination = final_destination

    # Return the start state.
    # States are your current airport and datetime
    def startState(self):
        return (self.origin, self.start_time)

    # Return set of actions possible from |state|.
    # Return all the flights you can take from an airport that are after your current time
    def actions(self, state):
        all_actions = []

        origin = state[0]
        today_tomorrow_flights = all_flights[origin][state[1].timetuple().tm_yday] + all_flights[origin][state[1].timetuple().tm_yday+1]
        for flight in today_tomorrow_flights:
            if hours_between(flight['departure_time'], state[1]) <= 24:
                # flight number, departure time, destination, real arrival time, elapsed time
                all_actions.append((flight[0],flight[1],flight[2],flight[3],flight[4]))
            else:
                break

        return all_actions

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (i.e. reaching the final destination airport) by setting the current airport to self.final_destination
    # * and returning an empty list (no more actions to take)
    def succAndProbReward(self, state, action):
        currentLocation = state[0]
        currentDatetime = state[1]

        flight_number = action[0]
        departure_time = action[1]
        destination = action[2]
        real_arrival_time = action[3]
        elapsed_time = action[4]

        if currentLocation == self.final_destination:
            return []
        else:
            # cancelled flight 
            succCancelled = (currentLocation, departure_time)
            probCancelled = 0.2
            rewardCancelled = 1/float((departure_time - currentDatetime).total_seconds())

            # regular flight, not cancelled
            succ = (destination, real_arrival_time)
            prob = 1
            reward = 1/float((departure_time - currentDatetime).total_seconds()) + 1/float(60*elapsed_time)

            return [(succCancelled, probCancelled, rewardCancelled), (succ, prob, reward)]

    def discount(self):
        return 1

mdp = FlightMDP(origin='EWR', start_time=datetime.datetime(2015, 10, 30, 8, 30), final_destination='SFO')
alg = util.ValueIteration()
alg.solve(mdp, .0001)

print alg.V
print alg.pi

############################################################
# Problem 4a: Q learning

# # Performs Q-learning.  Read util.RLAlgorithm for more information.
# # actions: a function that takes a state and returns a list of actions.
# # discount: a number between 0 and 1, which determines the discount factor
# # featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# # explorationProb: the epsilon value indicating how frequently the policy
# # returns a random action
# class QLearningAlgorithm(util.RLAlgorithm):
#     def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
#         self.actions = actions
#         self.discount = discount
#         self.featureExtractor = featureExtractor
#         self.explorationProb = explorationProb
#         self.weights = defaultdict(float)
#         self.numIters = 0

#     # Return the Q function associated with the weights and features
#     def getQ(self, state, action):
#         score = 0
#         for f, v in self.featureExtractor(state, action):
#             score += self.weights[f] * v
#         return score

#     # This algorithm will produce an action given a state.
#     # Here we use the epsilon-greedy algorithm: with probability
#     # |explorationProb|, take a random action.
#     def getAction(self, state):
#         self.numIters += 1
#         if random.random() < self.explorationProb:
#             return random.choice(self.actions(state))
#         else:
#             return max((self.getQ(state, action), action) for action in self.actions(state))[1]

#     # Call this function to get the step size to update the weights.
#     def getStepSize(self):
#         return 1.0 / math.sqrt(self.numIters)

#     # We will call this function with (s, a, r, s'), which you should use to update |weights|.
#     # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
#     # You should update the weights using self.getStepSize(); use
#     # self.getQ() to compute the current estimate of the parameters.
#     def incorporateFeedback(self, state, action, reward, newState):
#         # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
#         # raise Exception("Not implemented yet")
#         if newState != None:
#             v_opt = max((self.getQ(newState, action) for action in self.actions(newState)))
#             full_term = self.getStepSize()*(self.getQ(state, action) - (reward + self.discount*v_opt))
#             features = self.featureExtractor(state, action)
#             for feature in features:
#                 new_weight = full_term * feature[1]
#                 self.weights[feature[0]] -= new_weight
#         # END_YOUR_CODE

# # Return a single-element list containing a binary (indicator) feature
# # for the existence of the (state, action) pair.  Provides no generalization.
# def identityFeatureExtractor(state, action):
#     featureKey = (state, action)
#     featureValue = 1
#     return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
# smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# # Large test case
# largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

# def simulate_QL_over_MDP(mdp, featureExtractor):
#     # NOTE: adding more code to this function is totally optional, but it will probably be useful
#     # to you as you work to answer question 4b (a written question on this assignment).  We suggest
#     # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
#     # and then print some stats comparing the policies learned by these two approaches.
#     # BEGIN_YOUR_CODE
#     vi = util.ValueIteration()
#     vi.solve(mdp)
#     vlActions = {}
#     vlActions['Take'] = 0
#     vlActions['Peek'] = 0
#     vlActions['Quit'] = 0
#     for v in vi.pi.values():
#         vlActions[v] += 1
#     print 'VI = {}'.format(vlActions)

#     ql = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor)
#     util.simulate(mdp, ql, numTrials=30000, maxIterations=1000, verbose=False, sort=False)
#     ql.explorationProb = 0.0
#     mdp.computeStates()
#     actions = {}
#     actions['Take'] = 0
#     actions['Peek'] = 0
#     actions['Quit'] = 0
#     for state in mdp.states:
#         actions[ql.getAction(state)] += 1
#     print 'QL = {}'.format(actions)
    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
# def blackjackFeatureExtractor(state, action):
#     total, nextCard, counts = state

#     # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
#     all_features = []
#     # total feature
#     all_features.append(((action, total), 1))
#     # presence/absence
#     if counts != None:
#         presence = []
#         for i in counts:
#             if i > 0: presence.append(1)
#             else: presence.append(0)
#         all_features.append(((action, tuple(presence)), 1))
#         for j in range(len(counts)):
#             all_features.append(((action, j, counts[j]), 1))
#     return all_features
    # END_YOUR_CODE

# ############################################################
# # Problem 4d: What happens when the MDP changes underneath you?!

# # Original mdp
# originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# # New threshold
# newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

# def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
#     # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
#     # this partial function here to help you figure out the answer to 4d (a written question).
#     # Consider adding some code here to simulate two different policies over the modified MDP
#     # and compare the rewards generated by each.
#     # BEGIN_YOUR_CODE
#     vi = util.ValueIteration()
#     vi.solve(original_mdp)
#     opt_policy = vi.pi

#     fixedAlgo = util.FixedRLAlgorithm(opt_policy)
#     viRewards = util.simulate(modified_mdp, fixedAlgo, numTrials=30000, maxIterations=1000, verbose=False, sort=False)
#     print 'Average Value Iteration reward = {}'.format(sum(viRewards)/float(len(viRewards)))

#     ql = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor)
#     qlRewards = util.simulate(modified_mdp, ql, numTrials=30000, maxIterations=1000, verbose=False, sort=False)
#     print 'Average Q-learning reward = {}'.format(sum(qlRewards)/float(len(qlRewards)))
#     # END_YOUR_CODE