# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import queue

# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
from queue import PriorityQueue

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def getGreedyUpdate(self, state):
        """computes a one step-ahead value update and return it"""
        if self.mdp.isTerminal(state):
            return self.values[state]
        actions = self.mdp.getPossibleActions(state)
        vals = util.Counter()
        for action in actions:
            vals[action] = self.computeQValueFromValues(state, action)
        return max(vals.values())

    def runValueIteration(self):
        # Write value iteration code here
        # Modified to perform the iterative value update based on a greedy policy.
        # It iterates through each state and updates its value based on the maximum Q-value over possible actions.
        for _ in range(self.iterations):
            update_values_dict = util.Counter()
            for state in self.mdp.getStates():
                update_values_dict[state] = self.getGreedyUpdate(state)
            self.values = update_values_dict

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        #computes the Q-value by summing over possible transitions, using the reward and future value for each transition.
        q_value = 0

        possible_next_states = self.mdp.getTransitionStatesAndProbs(state, action)

        for nextState, prob in possible_next_states:
            reward = self.mdp.getReward(state, action, nextState)
            q_value += prob * (reward + self.discount * self.values[nextState])

        return q_value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #computes the optimal action by selecting the one with the highest Q-value in the given state.
        if self.mdp.isTerminal(state):
            return None
        action_values = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            action_values[action] = self.computeQValueFromValues(state, action)
        return action_values.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take a mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def setupAllPredecessors(self):
        #compute predecessors of all states and save it in a util.Counter() and return it
        # this function builds a dictionary of sets where each key (state) maps to all its predecessors.
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            predecessors[next_state].add(state)
        return predecessors

    def setupPriorityQueue(self):
        # setup priority queue for all states based on their highest diff in greedy update
        # This initializes a priority queue for prioritized sweeping.
        # it calculates the difference (diff) between the current value and the max Q-value for each state.
        # states with larger differences have higher priority in the queue.
        priority_queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max_q_value = max([self.computeQValueFromValues(state, action)
                                   for action in self.mdp.getPossibleActions(state)])
                diff = abs(self.values[state] - max_q_value)
                priority_queue.update(state, -diff)
        return priority_queue

    def runValueIteration(self):
        
        #compute predecessors of all states
        # we are gonna initialize predecessors and priority queue for prioritized sweeping.
        # states are updated in the order of priority, calculated by their value difference.
        allpreds = self.setupAllPredecessors()

        # setup priority queue
        pq = self.setupPriorityQueue()

        # run priority sweeping value iteration:
        # Updates the value of states based on prioritized order, using the threshold theta.
        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                possible_actions = self.mdp.getPossibleActions(state)
                q_values = []
                for action in possible_actions:
                    q_value = self.computeQValueFromValues(state, action)
                    q_values.append(q_value)
                self.values[state] = max(q_values)
            for predecessor in allpreds[state]:
                if not self.mdp.isTerminal(predecessor):
                    possible_actions = self.mdp.getPossibleActions(predecessor)
                    q_values = []
                    for action in possible_actions:
                        q_value = self.computeQValueFromValues(predecessor, action)
                        q_values.append(q_value)
                    max_q_value = max(q_values)
                    diff = abs(self.values[predecessor] - max_q_value)
                    if diff > self.theta:
                        pq.update(predecessor, -diff)
