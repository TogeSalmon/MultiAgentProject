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
        # score = 0
        # closesGhostPosition = newGhostStates[0].configuration.pos
        # closestGhost = manhattanDistance(newPos, closesGhostPosition)

        # newFoodPositions = newFood.asList()
        # foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions]
        # if len(foodDistances) == 0:
        #     return 0
        # closestFood = min(foodDistances)

        # if action == "Stop":
        #     score -= 50
        currentPos = list(successorGameState.getPacmanPosition())
        min = 999999999
        dist = 0
        currentFood = currentGameState.getFood()
        
        foodList = currentFood.asList()
        for i in range(len(foodList)):
            dist =  (manhattanDistance(foodList[i], currentPos))
            if dist < min:
                min = dist
        min = -min
        
        for state in newGhostStates:
            if state.scaredTimer == 0 and state.getPosition() == tuple(currentPos):
                return -999999999

        if action == 'Stop':
            return -999999999
        return min 

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
        def value(state, depth, agent_index):
            # print("Depth: ", depth)
            # print("self.depth: ", self.depth)
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if agent_index == 0:
                return max_value(state, depth, agent_index)
            else:
                return min_value(state, depth, agent_index)
            
        def max_value(state, depth, agent_index):
            v = -float('inf')
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                v = max(v, value(next_state, depth, agent_index + 1))
            return v
        
        def min_value(state, depth, agent_index):
            v = float('inf')
            num_agents = state.getNumAgents()
            next_agent_index = agent_index + 1
            next_depth = depth
            if next_agent_index >= num_agents:
                next_agent_index = 0
                next_depth += 1
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                v = min(v, value(next_state, next_depth, next_agent_index))
            return v
        
        legalMoves = gameState.getLegalActions(0)
        bestScore = -float('inf')
        bestAction = None
        for action in legalMoves:
            next_state = gameState.generateSuccessor(0, action)
            score = value(next_state, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
    
    #     action = self.minimax(0, 0, gameState)
    #     score = self.minimax(0, 0, gameState)
    #     #util.raiseNotDefined()
    #     return action
    
    # def minimax(self, curr_depth, agent_index, gameState):
    #     if agent_index >= gameState.getNumAgents():
    #         agent_index = 0
    #         curr_depth += 1
    #     if curr_depth == self.depth:
    #         return None, self.evaluationFunction(gameState)
        
    #     best_score = None
    #     best_action = None
    #     if agent_index == 0:
    #         for action in gameState.getLegalActions(agent_index):
    #             next_game_state = gameState.generateSuccessor(agent_index, action)
    #             score = self.minimax(curr_depth, agent_index+1, next_game_state)
    #             if best_score is None or score > best_score:
    #                 best_score = score
    #                 best_action = action
    #     else:
    #         for action in gameState.getLegalActions(agent_index):
    #             next_game_state = gameState.generateSuccessor(agent_index, action)

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(state, depth, agent_index, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return scoreEvaluationFunction(state)
            
            if agent_index == 0:
                return max_value(state, depth, agent_index, alpha, beta)
            else:
                return min_value(state, depth, agent_index, alpha, beta)
        
        def max_value(state, depth, agent_index, alpha, beta):
            v = -float('inf')
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                v = max(v, value(next_state, depth, agent_index + 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(state, depth, agent_index, alpha, beta):
            v = float('inf')
            num_agents = state.getNumAgents()
            next_agent_index = agent_index + 1
            next_depth = depth
            if next_agent_index >= num_agents:
                next_agent_index = 0
                next_depth += 1
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                v = min(v, value(next_state, next_depth, next_agent_index, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
        legalMoves = gameState.getLegalActions(0)
        bestScore = -float('inf')
        bestAction = None
        alpha = -float('inf')
        beta = float('inf')

        for action in legalMoves:
            next_state = gameState.generateSuccessor(0, action)
            score = value(next_state, 0, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction
        #util.raiseNotDefined()

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
        def value(state, depth, agent_index):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if agent_index == 0:
                return max_value(state, depth, agent_index)
            else:
                return exp_value(state, depth, agent_index)
            
        def max_value(state, depth, agent_index):
            v = -float('inf')
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                v = max(v, value(next_state, depth, agent_index + 1))
            return v
        
        def exp_value(state, depth, agent_index):
            v = 0
            num_agents = state.getNumAgents()
            next_agent_index = agent_index + 1
            next_depth = depth
            if next_agent_index >= num_agents:
                next_agent_index = 0
                next_depth += 1
            num_actions = len(state.getLegalActions(agent_index))
            prob = 1.0 / num_actions
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                v += prob * value(next_state, next_depth, next_agent_index)
            return v
        
        legalMoves = gameState.getLegalActions(0)
        bestScore = -float('inf')
        bestAction = None

        for action in legalMoves:
            next_state = gameState.generateSuccessor(0, action)
            score = value(next_state, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
