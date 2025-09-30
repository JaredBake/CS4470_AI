"""multiAgents.py - Multi-Agent Search Algorithms for Pacman
===========================================================

This module implements various multi-agent search algorithms for the Pacman game,
including reflex agents, minimax, alpha-beta pruning, and expectimax search.

The module provides agent classes that:
- Make decisions based on state evaluation functions
- Implement adversarial search algorithms
- Model both deterministic and probabilistic opponent behavior
- Search to configurable depths using evaluation heuristics

Key Classes:
    ReflexAgent: Makes decisions using state evaluation heuristics
    MinimaxAgent: Implements minimax search algorithm
    AlphaBetaAgent: Implements alpha-beta pruning search
    ExpectimaxAgent: Implements expectimax probabilistic search

Usage:
    This module is used by the Pacman game to create AI agents. Agents can be
    selected and configured via command line arguments.

Author: George Rudolph
Date: 14 Nov 2024
Major Changes:
1. Added type hints throughout the codebase for better code clarity and IDE support
2. Improved docstrings with detailed descriptions and Args/Returns sections
3. Enhanced code organization with better function and variable naming

This code runs on Python 3.13

Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import random, math
import util
from util import manhattanDistance
from game import Agent, Directions
from typing import List, Tuple, Any
from pacman import GameState

class ReflexAgent(Agent):
    """A reflex agent that chooses actions by examining alternatives via a state evaluation function.
    
    This agent evaluates each possible action using a heuristic evaluation function and selects
    among the best options. The evaluation considers factors like:
    - Distance to ghosts (avoiding them)
    - Score improvements
    - Distance to food
    - Maintaining movement direction
    """

    def getAction(self, gameState: GameState) -> str:
        """Choose among the best actions according to the evaluation function.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: A direction from Directions.{North, South, West, East, Stop}
            
        The method collects legal moves, scores them using the evaluation function,
        and randomly selects among those with the best score.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
        """Evaluate the desirability of a game state after taking an action.
        
        Args:
            currentGameState: The current game state
            action: The proposed action
            
        Returns:
            float: A score where higher numbers are better, using values 8,4,2,1,0
            that are bitwise orthogonal (powers of 2)
            
        The function evaluates states based on:
        - Avoiding ghosts (returns 0 if too close)
        - Score improvements (returns 8)
        - Getting closer to food (returns 4) 
        - Maintaining direction (returns 2)
        - Default case (returns 1)
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        minDist = 9999
        minFood = None
        if newFood.count() == 0:
            return 4
        for food in newFood.asList():
            if manhattanDistance(newPos, food) < minDist:
                minFood = food
                minDist = manhattanDistance(newPos, food)
        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) <= 1:
                return 0
        if manhattanDistance(newPos, minFood) == 0:
            return 8
        if manhattanDistance(currentGameState.getPacmanPosition(), minFood) > manhattanDistance(newPos, minFood):
            return 4
        if manhattanDistance(currentGameState.getPacmanPosition(), minFood) == manhattanDistance(newPos, minFood):
            return 2
        return 2


def scoreEvaluationFunction(currentGameState: GameState) -> float:
    """Return the score of the state for use with adversarial search agents.
    
    Args:
        currentGameState: The game state to evaluate
        
    Returns:
        float: The score displayed in the Pacman GUI
        
    This is the default evaluation function for adversarial search agents.
    Not intended for use with reflex agents.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """Base class for adversarial search agents (minimax, alpha-beta, expectimax).
    
    This abstract class provides common functionality for multi-agent searchers.
    It should not be instantiated directly, but rather extended by concrete
    agent implementations.
    
    Attributes:
        index: Agent index (0 for Pacman)
        evaluationFunction: Function used to evaluate game states
        depth: Maximum depth of search tree
    """

    def __init__(self, evalFn: str = 'scoreEvaluationFunction', depth: str = '2') -> None:
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """Minimax agent that implements adversarial search.
    
    This agent uses minimax search to determine the optimal action by considering
    the worst case scenario at each level.
    """

    def getAction(self, gameState: GameState) -> str:

        numAgents = gameState.getNumAgents()

        def minimax(state: GameState, agentIndex: int, currentDepth: int) -> float:
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

            if agentIndex == 0:  # Pacman
                bestVal = -float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, nextAgent, nextDepth)
                    if val > bestVal:
                        bestVal = val
                return bestVal
            else:  # Ghosts
                bestVal = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, nextAgent, nextDepth)
                    if val < bestVal:
                        bestVal = val
                return bestVal

        # Root
        legalMoves = gameState.getLegalActions(0)
        if not legalMoves:
            return Directions.STOP

        bestScore = -float('inf')
        bestAction = legalMoves[0]
        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            nextAgent = 1 % numAgents
            nextDepth = 0 + (1 if nextAgent == 0 else 0)
            score = minimax(successor, nextAgent, nextDepth)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Minimax agent with alpha-beta pruning optimization.
    
    This agent implements minimax search with alpha-beta pruning to more efficiently
    explore the game tree by pruning branches that cannot affect the final decision.
    """

    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action using alpha-beta pruning.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: The optimal action according to alpha-beta pruning
            
        Pacman is always the max agent, ghosts are always min agents.
        At depth 0, max_value returns an action. At other depths, it returns a value.
        """

        # Multi-agent alpha-beta search
        numAgents = gameState.getNumAgents()

        def alphabeta(state: GameState, agentIndex: int, currentDepth: int, alpha: float, beta: float) -> float:
            # Terminal or depth cutoff
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

            if agentIndex == 0:  # Maximizer (Pacman)
                value = -float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(successor, nextAgent, nextDepth, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else: 
                value = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(successor, nextAgent, nextDepth, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        legalMoves = gameState.getLegalActions(0)
        if not legalMoves:
            return Directions.STOP

        bestScore = -float('inf')
        bestAction = legalMoves[0]
        alpha = -float('inf')
        beta = float('inf')
        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = alphabeta(successor, 1 % numAgents, 0 + (1 if (1 % numAgents) == 0 else 0), alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """An agent that uses expectimax search to make decisions.
    
    This agent models ghosts as choosing uniformly at random from their legal moves.
    It uses expectimax search to find optimal actions against probabilistic opponents.
    
    The agent searches to a fixed depth using a supplied evaluation function.
    """

    def getAction(self, gameState: GameState) -> str:
        """Return the expectimax action using self.depth and self.evaluationFunction.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: The selected action (one of Directions.{North,South,East,West,Stop})
            
        All ghosts are modeled as choosing uniformly at random from their legal moves.
        """
        numAgents = gameState.getNumAgents()

        def expectimax(state: GameState, agentIndex: int, currentDepth: int) -> float:
            # Terminal or depth cutoff
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

            if agentIndex == 0:  # Pacman (maximizer)
                bestVal = -float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = expectimax(successor, nextAgent, nextDepth)
                    if val > bestVal:
                        bestVal = val
                return bestVal
            else:  # Ghosts (expected value)
                total = 0.0
                prob = 1.0 / len(legalActions)
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    total += prob * expectimax(successor, nextAgent, nextDepth)
                return total

        # Root: choose the action with highest expectimax value
        legalMoves = gameState.getLegalActions(0)
        if not legalMoves:
            return Directions.STOP

        bestScore = -float('inf')
        bestAction = legalMoves[0]
        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, 1 % numAgents, 0 + (1 if (1 % numAgents) == 0 else 0))
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(game_state: GameState) -> float:
    """A more sophisticated evaluation function for Pacman game states.
    
    This function evaluates states by combining the game score with a penalty
    based on distance to the closest food pellet. The penalty uses the reciprocal
    of the distance to give higher penalties to food that is farther away.
    
    Args:
        game_state: The game state to evaluate
        
    Returns:
        float: The evaluation score where higher values are better
    """
    
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
# Abbreviation
better = betterEvaluationFunction

