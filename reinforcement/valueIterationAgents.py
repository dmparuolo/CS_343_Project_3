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
        self.visitedTiles = set()   # A set of tiles that have already been visited
        self.newTiles = set() # A set of new tiles that need to be visited 
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(1, self.iterations + 1):
            # print(f"Iteration {i}:")

            batchValues = util.Counter()


            for state in self.mdp.getStates():

                if self.mdp.isTerminal(state):
                    batchValues[state] = 0
                    continue

                maxQ = None
                for action in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, action)
                    
                    if maxQ == None or q > maxQ:
                        maxQ = q
                
                # print(f"\tState: {state}, Max Q: {maxQ}")
                
                maxQ = 0 if maxQ == None else maxQ
                batchValues[state] = maxQ
            
            self.values = batchValues.copy()

            # print(f"\tValues: {self.values}")

            


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
        "*** YOUR CODE HERE ***"
        sum = 0
        t = self.mdp.getTransitionStatesAndProbs(state, action) # list of (state, prob)
        for next in t: 
            r = self.mdp.getReward(state, action, next[0])

            sum += next[1]*(r + (self.discount * self.getValue(next[0])))
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        

        actions = self.mdp.getPossibleActions(state)

        if not actions:
            return None
        
        maxAction = max(actions, key=lambda a: self.computeQValueFromValues(state, a))
        return maxAction

            
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        s = 0
        for i in range(self.iterations):

            state = self.mdp.getStates()[s]
            # print(self.mdp.getStates())
            
            if self.mdp.isTerminal(state):
                self.values[state] = 0
                s += 1
                if s == len(self.mdp.getStates()):
                    s = 0
                continue

            maxQ = None
            for action in self.mdp.getPossibleActions(state):
                q = self.computeQValueFromValues(state, action)
                
                if maxQ == None or q > maxQ:
                    maxQ = q
            
            # print(f"\tState: {state}, Max Q: {maxQ}")
            
            maxQ = 0 if maxQ == None else maxQ
            self.values[state] = maxQ

            s += 1
            if s == len(self.mdp.getStates()):
                s = 0


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        predecessors = self.getAllPredecessors()
        
        queue = util.PriorityQueue()

        for state in self.mdp.getStates():

            if self.mdp.isTerminal(state):
                continue
            
            maxQ = None
            for action in self.mdp.getPossibleActions(state):
                q = self.computeQValueFromValues(state, action)
                
                if maxQ == None or q > maxQ:
                    maxQ = q

            diff = abs(self.values[state] - maxQ)
            queue.update(state, -diff)
            
        for i in range(self.iterations):

            if queue.isEmpty():
                break

            s = queue.pop()

            if not self.mdp.isTerminal(s):
                maxQ = None
                for action in self.mdp.getPossibleActions(s):
                    q = self.computeQValueFromValues(s, action)
                    
                    if maxQ == None or q > maxQ:
                        maxQ = q
                self.values[s] = maxQ

            for p in predecessors[s]:

                if self.mdp.isTerminal(p):
                    continue

                maxQ = None
                for action in self.mdp.getPossibleActions(p):
                    q = self.computeQValueFromValues(p, action)
                    
                    if maxQ == None or q > maxQ:
                        maxQ = q
                
                diff = abs(self.values[p] - maxQ)
                if diff > self.theta:
                    queue.update(p, -diff)
    

    def getAllPredecessors(self):
        # predecessor of a state s is all states that have a nonzero prob of reaching s by taking action a
        predecessors = {}
        
        for state in self.mdp.getStates():

            if self.mdp.isTerminal(state):
                continue
            
            actions = self.mdp.getPossibleActions(state)

            for action in actions: 
                next = self.mdp.getTransitionStatesAndProbs(state, action) # (next state, prob)
                
                for n, p in next:
                    if n not in predecessors:
                        predecessors[n] = set()
                    if (p != 0):
                        predecessors[n].add(state)

        return predecessors 
