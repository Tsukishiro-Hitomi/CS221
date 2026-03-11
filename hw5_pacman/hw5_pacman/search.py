"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in search_agents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def get_start_state(self):
     """
     Returns the start state for the search problem 
     """
     util.raise_not_defined()
    
  def is_goal_state(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raise_not_defined()

  def get_successors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, step_cost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'step_cost' is the incremental 
     cost of expanding to that successor
     """
     util.raise_not_defined()

  def get_cost_of_actions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raise_not_defined()
           

def tiny_maze_search(problem):
  """
  Returns a sequence of moves that solves tiny_maze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tiny_maze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depth_first_search(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.get_start_state()
  print "Is the start a goal?", problem.is_goal_state(problem.get_start_state())
  print "Start's successors:", problem.get_successors(problem.get_start_state())
  """

def breadth_first_search(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
      
def uniform_cost_search(problem):
  "Search the node of least total cost first. "

def null_heuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def a_star_search(problem, heuristic=null_heuristic):
  "Search the node that has the lowest combined cost and heuristic first."
    
  
# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search