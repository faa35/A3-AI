# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent


import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # set all (state, action) pair to 0
        self.qvalues = util.Counter()  # A Counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Retrieves the Q-value for a state-action pair from self.qvalues
        # If the pair is not in self.qvalues, it returns the default value of 0.0
        return self.qvalues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # lets get the list of legal actions for the current state
        legal_actions = self.getLegalActions(state)
        if not legal_actions:  # Checks for terminal state with no legal actions
            return 0.0

        #we are calculating the maximum Q-value for all legal actions
        q_values = []  #initializing an empty list to store Q-values
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            q_values.append(q_value)

        return max(q_values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # gets the list of legal actions for the current state
        legal_actions = self.getLegalActions(state)
        if not legal_actions:  # Checks for terminal state with no legal actions
            return None

        # Identifies the action with the highest Q-value
        best_value = self.computeValueFromQValues(state)
        #Find all actions with the best Q-value.
        best_actions = []  # Initialize an empty list to store the best actions
        for action in legal_actions:
            q_value = self.getQValue(state, action)  
            if q_value == best_value:  
              best_actions.append(action)
        # In case of ties (multiple actions with the same Q-value), a random choice is made
        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Gets the list of legal actions for the current state
        legal_actions = self.getLegalActions(state)
        action = None

        if not legal_actions:  
        # No legal actions, so return None for terminal state
            return None
        # Uses epsilon-greedy policy: chooses a random action with probability epsilon
        if util.flipCoin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            # Otherwise, chooses the best action based on Q-values
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        """
              QLearning update algorithm:

              Q(s,a) = (1-alpha)*Q(s,a) + alpha*sample

              ***sample = R(s,a,s') + gamma*max(Q(s',a'))***

        """
        # Computes the sample value based on the observed transition
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        # Updates the Q-value for the (state, action) pair
        updated_q_value = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
        self.qvalues[(state, action)] = updated_q_value



    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

