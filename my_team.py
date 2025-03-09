# baseline_team.py
# ---------------
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

from sympy.codegen.ast import continue_

import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='BountyHunter', second='BorderPatrol', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class AlphaBetaAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.depth = 2

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):

        """
        Returns the minimax action using self.depth and self.evaluate
        """
        inf = float('inf')
        def value(state, depth, agent_idx, alpha, beta):
            #print(agent_idx)
            curr_agent_idx = agent_idx % state.get_num_agents()
            if depth == self.depth or state.is_over():
                return self.evaluate(state), None

            if curr_agent_idx in self.get_team(state):
                return max_value(state,depth , agent_idx, alpha, beta)

            elif state.get_agent_position(curr_agent_idx) is None:
                return value(state, depth, agent_idx + 1, alpha, beta)

            else:
                return min_value(state,depth , agent_idx, alpha, beta)

        def min_value(state, depth, agent_idx, alpha, beta):
            v = inf
            best_action = None
            idx = agent_idx % state.get_num_agents()
            # Get the successor for every legal action

            for action in state.get_legal_actions(idx):
                successor = state.generate_successor(idx, action)

                # If last ghost, increase depth and start with pacman again, otherwise go to the next ghost
                #next_depth = depth + 1 if idx % state.get_num_agents == self.index else depth
                if action == Directions.STOP:
                    continue
                newVal,_ = value(successor, depth , agent_idx + 1, alpha, beta)

                # takes lowest value and its action between current value and the new value
                if newVal < v:
                    v = newVal
                    best_action = action

                # Change beta to smallest value
                beta = min(beta, newVal)
                # Prune unnecessary branches
                if alpha > beta:
                    break
            return v, best_action

        def max_value(state, depth, agent_idx, alpha, beta):
            v = -inf
            best_action = None
            idx = agent_idx % state.get_num_agents()
            # Get the successor for every legal action

            for action in state.get_legal_actions(idx):
                successor = state.generate_successor(idx, action)
                if action == Directions.STOP:
                    continue
                next_depth = depth + 1 if (agent_idx % state.get_num_agents()) == self.index else depth
                newVal,_ = value(successor, next_depth, agent_idx + 1, alpha, beta)

                # takes lowest value and its action between current value and the new value
                if newVal > v:
                    v = newVal
                    best_action = action

                # change alpha to largest value
                alpha = max(alpha, newVal)
                # prune unnecessary branches
                if alpha > beta:
                    break
            return v, best_action

        bestAction = value(game_state,0 , self.index, -inf, inf)
        # return best action
        return bestAction[1]

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state)
        weights = self.get_weights(game_state)
        return features * weights

    def get_features(self, game_state):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()

        features['successor_score'] = self.get_score(game_state)
        return features

    def get_weights(self, game_state):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)



    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        print(features*weights)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class BountyHunter(AlphaBetaAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state):
        features = util.Counter()

        food_list = self.get_food(game_state).as_list()
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True, but better safe than sorry
            my_pos = game_state.get_agent_position(self.index)
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # How much food have I eaten
        features['eaten_food'] = my_state.num_carrying

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position is not None]
        features['num_defenders'] = len(defenders)
        features['num_attackers'] = 2 - len(defenders)

        if len(defenders) > 0:
            dists = game_state.agent_distances
            features['invader_distance'] = min(dists)
        for a in defenders:
            if a.scared_timer > 0:
                features['invader_distance'] = -features['invader_distance']
            else:
                features['invader_distance'] = 5 * features['invader_distance']
        if self.red:
            my_capsules = game_state.get_blue_capsules()
        else:
            my_capsules = game_state.get_red_capsules()
        if len(my_capsules) > 0:
            dists = [self.get_maze_distance(my_pos, p) for p in my_capsules]
            features['powerPellet'] = min(dists)



        defending_food = self.get_food_you_are_defending(game_state).as_list()
        if defending_food:
            closest_food = min([self.get_maze_distance(my_pos, food) for food in defending_food])
        else:
            closest_food = float('inf')
        features["return_distance"] = closest_food

        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        return features

    def get_weights(self, game_state):
        features = self.get_features(game_state)
        if features['eaten_food'] < 5:
            return {'successor_score': pow(0.65, 0.5) * 5,
                    'distance_to_food': -pow(0.65, 2),
                    'eaten_food': 10,
                    'invader_distance': -50,
                    'num_defenders': -10,
                    'num_attackers': 10,
                    'powerPellet': -2,
                    'return_distance': -0.01,
                    'on_defense': 0}
        elif features['eaten_food'] > 15:
            return {'successor_score': pow(0.65, 0.5) * 5,
                    'distance_to_food': -0.002,
                    'eaten_food': 2,
                    'invader_distance': -100,
                    'num_defenders': -40,
                    'num_attackers': 10,
                    'powerPellet': 0,
                    'return_distance': -50,
                    'on_defense': 1000}
        else:
            return {'successor_score': pow(0.65, 0.5) * 5,
                    'distance_to_food': -pow(0.65, 2),
                    'eaten_food': 2,
                    'invader_distance': -80,
                    'num_defenders': -30,
                    'num_attackers': 10,
                    'powerPellet': -4,
                    'return_distance': -1,
                    'on_defense': 10}


class BorderPatrol(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        # food needs defending (self added)
        defending_food = self.get_food_you_are_defending(successor).as_list()
        closest_food =  min([self.get_maze_distance(my_pos, food) for food in defending_food])

        # hold the line
        border_line = game_state.data.layout.width // 2
        boundary_x = border_line - 1 if self.red else border_line
        border_distance = abs(my_pos[0] - boundary_x)

        # camp own capsules
        capsules = self.get_capsules_you_are_defending(successor)
        closest_capsule = 0
        if capsules:
            closest_capsule = min(self.get_maze_distance(my_pos, capsules) for capsules in capsules)

        # go hunt if scared
        scary = my_state.scared_timer
        if scary > 0:
            return BountyHunter.get_features(successor)

        # set features depended on if enemy attacking
        if len(invaders) > 0:
            features['food_defense'] = 0
            if not capsules:
                features['near_border'] = closest_food
            else:
                features['near_border'] = 0
            features['capsule_defense'] = closest_capsule
        else:
            features['food_defense'] = 0
            features['near_border'] = border_distance
            features['capsule_defense'] = 0

        return features

    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        scary = my_state.scared_timer
        if scary==0:
            return {'num_invaders': -1000,
                'on_defense': 1000,
                'invader_distance': -100,
                'stop': -100, 'reverse': -2,
                'food_defense': -100,
                'near_border' : -100,
                'capsule_defense': -100}
        else:
            return BountyHunter.get_weights(successor)

                # {'num_invaders': 0,
                # 'on_defense': 1,
                # 'invader_distance': 100,
                # 'stop': -100, 'reverse': -2,
                # 'food_defense': 0,
                # 'near_border' : 0,
                # 'capsule_defense': 0}