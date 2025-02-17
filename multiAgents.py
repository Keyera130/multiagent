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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # Avoid stopping
        if action == Directions.STOP:
            return float('-inf')

        # Get the Manhattan distances to all food pellets
        foodList = newFood.asList()
        foodDistances = [util.manhattanDistance(newPos, food) for food in foodList]

        # Get the Manhattan distances to all ghosts
        ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        # Closest food score (inverse distance, prioritize eating food)
        foodScore = 0
        if foodDistances:
            foodScore = 1 / min(foodDistances)  # Closer food gives higher score

        # Closest ghost score (negative impact if too close)
        ghostScore = 0
        for i in range(len(ghostDistances)):
            if newScaredTimes[i] > 0:  # If the ghost is scared, prioritize eating it
                ghostScore += 10 / (ghostDistances[i] + 1)
            else:  # Otherwise, avoid the ghost
                if ghostDistances[i] < 2:  # Immediate danger
                    ghostScore -= 10

        # Return a weighted sum of factors
        return successorGameState.getScore() + (10 * foodScore) + ghostScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agentIndex):
            # Base case: Terminal state (win/lose) or max depth reached
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Pacman (Maximizing Player)
            if agentIndex == 0:
                return max(
                    (minimax(state.generateSuccessor(agentIndex, action), depth, 1)
                     for action in state.getLegalActions(agentIndex)),
                    default=float('-inf')
                )

            # Ghosts (Minimizing Players)
            else:
                nextAgent = agentIndex + 1  # Move to the next ghost
                if nextAgent >= state.getNumAgents():  # If last ghost, go to Pacman and increase depth
                    nextAgent, nextDepth = 0, depth + 1
                else:
                    nextDepth = depth

                return min(
                    (minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                     for action in state.getLegalActions(agentIndex)),
                    default=float('inf')
                )

        # Choose action corresponding to the max-value successor
        bestAction = max(
            gameState.getLegalActions(0),
            key=lambda action: minimax(gameState.generateSuccessor(0, action), 0, 1)
        )
        return bestAction
       

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Base Case: Terminal state or depth limit
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Pacman (Maximizing)
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None

                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = alphaBeta(successor, depth, 1, alpha, beta)

                    if value > bestValue:
                        bestValue, bestAction = value, action

                    alpha = max(alpha, bestValue)
                    
                    if bestValue > beta:  # Only prune when strictly greater
                        break

                return bestValue if depth > 0 else bestAction

            # Ghosts (Minimizing)
            else:
                bestValue = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth

                if nextAgent >= state.getNumAgents():  # Last ghost -> Next turn for Pacman
                    nextAgent = 0
                    nextDepth += 1

                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)

                    bestValue = min(bestValue, value)
                    beta = min(beta, bestValue)

                    if bestValue < alpha:  # Only prune when strictly less
                        break

                return bestValue

        # Root call: Must return an action
        return alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            # Base Case: Terminal state or depth limit
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Pacman (Maximizing)
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None

                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimax(successor, depth, 1)

                    if value > bestValue:
                        bestValue, bestAction = value, action

                return bestValue if depth > 0 else bestAction

            # Ghosts (Chance Nodes)
            else:
                actions = state.getLegalActions(agentIndex)
                numActions = len(actions)

                if numActions == 0:
                    return self.evaluationFunction(state)

                totalValue = 0

                nextAgent = agentIndex + 1
                nextDepth = depth

                if nextAgent >= state.getNumAgents():  # Last ghost -> Pacman's turn
                    nextAgent = 0
                    nextDepth += 1

                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimax(successor, nextDepth, nextAgent)
                    totalValue += value

                return totalValue / numActions  # Compute expectation

        # Root call: Must return an action
        return expectimax(gameState, 0, 0)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Basic score (current score)
    score = currentGameState.getScore()

    # Food distance: We want Pacman to eat food that is closest.
    foodDist = [manhattanDistance(pacmanPos, food) for food in foodList]
    minFoodDist = min(foodDist) if foodDist else 0  # Avoid errors if there's no food
    foodScore = -minFoodDist  # Negative because we want Pacman closer to food

    # Ghost distance: Penalize if Pacman is too close to a ghost.
    ghostScores = []
    for i, ghostState in enumerate(ghostStates):
        ghostDist = manhattanDistance(pacmanPos, ghostState.getPosition())
        if scaredTimes[i] > 0:
            ghostScores.append(ghostDist)  # If ghost is scared, we want to get closer
        else:
            ghostScores.append(-ghostDist)  # Otherwise, avoid ghosts
    ghostScore = sum(ghostScores)

    # Return final evaluation: Combine the features
    return score + foodScore + ghostScore

# Abbreviation
better = betterEvaluationFunction
