# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_distance_list = []
        ghost_distance_list = []
        food_list = currentGameState.getFood().asList()

        # Find the nearest ghost to the Pac-Man
        for ghost in newGhostStates:
            # Verify that the ghost's position will not be the same as Pac-Man
            if ghost.getPosition() == newPos:
                # Verify if it's not a frozen ghost
                if not (0 in newScaredTimes):
                    ghost_distance_list.append(util.manhattanDistance(ghost.getPosition(), newPos))
                else:
                    return -999999
            else:
                # Add the distance to the list to maintain a list of manhattan distances between the ghosts
                # and Pac-Man's successor position
                ghost_distance_list.append(util.manhattanDistance(newPos, ghost.getPosition()))

        # Return the least distance from the list i.e. this will return the nearest ghost
        nearest_ghost=min(ghost_distance_list)

        # Find the nearest food to the Pac-Man
        for food in food_list:
            # Add the distance to the list to maintain a list of manhattan distances between the food
            # and Pac-Man's successor position
            food_distance_list.append((util.manhattanDistance(food, newPos)))

        # Return the least distance from the list i.e. this will return the nearest food
        nearest_food=min(food_distance_list)

        # Create the evaluation function by assigning weights to the nearest food & the ghost.
        # If the Pac-Man is very near to the ghost(lesser than 2 places), abort mission. If not, more weightage
        # is given to the food.
        if nearest_ghost < 2:
            return -9999999
        else:
            result_score = 1/( nearest_food* 0.9 + nearest_ghost * 0.1)
            return result_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # The game begins with the maximising player function being called
        result_score, result_action = self.pacman_minimax(gameState, 0, self.depth)
        # Return the legal result_action
        return result_action
    # A max function for Pac-Man, who is our maximising agent
    def pacman_minimax(self, gameState, agent_type, depth):
        successors=[]
        maximum=-99999

        # Check if it's a terminal state. Return the score if it is.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"

        # It's Pac-Man!
        if agent_type==0:
            # Append successors to the successor list for all legal actions
            # for Pac-Man
            for action in gameState.getLegalActions(agent_type):
                # The successor list will have a tuple of (score, action) and the action
                successors.append((self.ghost_minimax(gameState.generateSuccessor(agent_type, action), 1, depth),action))
            # Get the maximum score and its associated action from the successors list

            for succ_tuple,action in successors:
                i=succ_tuple[0]
                if i>maximum:
                    maximum=i
                    result_action = action

            # Return the maximum score and the associated action
            return maximum,result_action

    # A min function for a ghost, who is our minimising agent
    def ghost_minimax(self, gameState, agent_type, depth):
        successors=[]
        minimum=99999

        # Check if it's a terminal state. Return the score if it is.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"

        # Check if it's the last ghost. If it is, it will be Pac-Man's turn next
        if agent_type == gameState.getNumAgents() - 1:
            # Append successors to the successor list for all legal actions
            # for an agent
            for action in gameState.getLegalActions(agent_type):
                # The successor list will have a tuple of (score, action) and the action
                successors.append((self.pacman_minimax(gameState.generateSuccessor(agent_type, action), 0, depth - 1), action))
        else:
            for action in gameState.getLegalActions(agent_type):
                successors.append((self.ghost_minimax(gameState.generateSuccessor(agent_type, action), agent_type + 1, depth), action))
        # Get the minimum score and its associated action from the successors list
        for succ_tuple, j in successors:
            i = succ_tuple[0]
            if i < minimum:
                minimum = i
                result_action = j

        # Return the minimum score and the associated action for that ghost
        return minimum, result_action

        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # The game begins with the maximising player function being called
        result_score, result_action = self.pacman_minimax(gameState, 0, self.depth,-99999,99999)
        # Return the legal result_action
        return result_action

    # A max function for Pac-Man, who is our maximising agent
    def pacman_minimax(self, gameState, agent_type, depth,alpha,beta):
        maximum = -99999

        # Check if it's a terminal state. Return the score if it is.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"

        # It's Pac-Man!
        if agent_type == 0:
            for action in gameState.getLegalActions(agent_type):
                successors = self.ghost_minimax(gameState.generateSuccessor(agent_type, action), 1, depth,alpha,beta), action
                # Store the value of the successor node
                score=successors[0][0]
                # The alpha value will be the highest of the current alpha and the score
                alpha = max(alpha, score)
                if score > maximum:
                    maximum = score
                    result_action = action
                # If the maximum value is more than the beta, prune the tree
                if maximum > beta:
                    break
            # return the maximum value and the associated action
            return maximum, result_action

    # A min function for a ghost, who is our minimising agent
    def ghost_minimax(self, gameState, agent_type, depth,alpha,beta):
        minimum = 99999

        # Check if it's a terminal state. Return the score if it is.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"

        # Handle the last ghost condition or the number of nodes expanded will go wrong
        if agent_type == gameState.getNumAgents() - 1:
            for action in gameState.getLegalActions(agent_type):
                successors=self.pacman_minimax(gameState.generateSuccessor(agent_type, action), 0, depth - 1,alpha,beta), action
                # Store the value of the successor node
                score=successors[0][0]
                # The beta value will be the lower one between the current beta and the score
                beta = min(beta, score)
                # Keep track of the minimum score among all successors and the associated action
                if score < minimum:
                    minimum = score
                    result_action = action
                # If the minimum value is less than the alpha, prune the tree
                if alpha > minimum:
                    break
        else:
            for action in gameState.getLegalActions(agent_type):
                successors= self.ghost_minimax(gameState.generateSuccessor(agent_type, action), agent_type + 1,depth,alpha,beta), action
                # Store the value of the successor node
                score = successors[0][0]
                # The beta value will be the lower one between the current beta and the score
                beta = min(beta, score)
                # Keep track of the minimum score among all successors and the associated action
                if score < minimum:
                    minimum = score
                    result_action = action
                # If the minimum value is less than the alpha, prune the tree
                if alpha > minimum:
                    break
        # return the minimum value and the associated action
        return minimum, result_action

        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        result_score, result_action = self.pacman_minimax(gameState, 0, self.depth)
        return result_action

    def pacman_minimax(self, gameState, agent_type, depth):
        successors = []
        maximum = -99999

        # Check if it's a terminal state. Return the score if it is.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"

        # It's Pac-Man!
        if agent_type == 0:
            for action in gameState.getLegalActions(agent_type):
                successors=self.ghost_minimax(gameState.generateSuccessor(agent_type, action), 1, depth), action
                # Store the value of the successor node
                score = successors[0][0]
                # Keep track of the maximum score among all successors and the associated action
                if score > maximum:
                    maximum = score
                    result_action = action

            # return the maximum value and the associated action
            return maximum, result_action

    def ghost_minimax(self, gameState, agent_type, depth):
        successors = []
        minimum = 99999

        # Check if it's a terminal state. Return the score if it is.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"

        # Check if it's the last ghost. If it is, it will be Pac-Man's turn next
        if agent_type == gameState.getNumAgents() - 1:
            # Variables to find the expected value at a chance node
            p = 1.0 / len(gameState.getLegalActions(agent_type))
            a = 0
            for action in gameState.getLegalActions(agent_type):
                successors=self.pacman_minimax(gameState.generateSuccessor(agent_type, action), 0, depth - 1), action
                # Store the value of the successor node
                score = successors[0][0]
                # Find the expected value of the chance node
                a=a+p*(score)
                # Store the associated action
                result_action=action
        else:
            # Variables to find the expected value at a chance node
            p = 1.0 / len(gameState.getLegalActions(agent_type))
            a = 0
            for action in gameState.getLegalActions(agent_type):
                successors=self.ghost_minimax(gameState.generateSuccessor(agent_type, action), agent_type + 1,depth), action
                # Store the value of the successor node
                score = successors[0][0]
                # Find the expected value of the chance node
                a = a + p * (score)
                # Store the associated action
                result_action = action

        # return the expected(probabilistic) value and the associated action
        return a, result_action

        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

