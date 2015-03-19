from __future__ import print_function
from __future__ import division

import numpy as np
import random
import functools

import logging

__author__ = 'johannes'


class State(object):
    def perform(self, action):
        return self

    def is_terminal(self):
        return False

    @property
    def actions(self):
        return [0]

    @actions.setter
    def actions(self, actions):
        raise ValueError("Actions can not be set.")

    def __eq__(self, other):
        return False

    def __hash__(self):
        return 1

    def reward(self, parent, action):
        return 0


class Node(object):
    def __init__(self, parent):
        self.parent = parent
        self.children = {}
        self.q = 0
        self.n = 0


class StateNode(Node):

    def __init__(self, parent, state, reward):
        super(StateNode, self).__init__(parent)
        self.state = state
        self.reward = reward
        for action in state.actions:
            self.children[action] = ActionNode(self, action)

    @property
    def untried_actions(self):
        return [a for a in self.children if self.children[a].n == 0]

    @untried_actions.setter
    def untried_actions(self, value):
        raise ValueError("Actions can not be set.")

    def __str__(self):
        return "State: {}".format(self.state)


class ActionNode(Node):
    def __init__(self, parent, action):
        super(ActionNode, self).__init__(parent)
        self.action = action
        self.n = 0

    def sample_state(self, real_world=False):
        if real_world:
            state = self.parent.state.real_world_perform(self.action)
        else:
            state = self.parent.state.perform(self.action)

        if state not in self.children:
            r = state.reward(self.parent.state, self.action)
            self.children[state] = StateNode(self, state, r)

        if real_world:
            self.children[state].state.belief = state.belief

        return self.children[state]

    def __str__(self):
        return "Action: {}".format(self.action)


def uct_search(root, gamma, n=1500):
    logger = logging.getLogger('uct')
    for i in range(n):
        print('.', end='')
        node = tree_policy(root)
        bellman_backup(node, gamma)

    logger.debug(dict([(action.action, action.q)
                       for action in root.children.values()]))
    return best_child(root, 0).parent.action


def tree_policy(state_node):
    while not state_node.state.is_terminal():
        if state_node.untried_actions:
            return expand(state_node)
        else:
            state_node = best_child(state_node, np.sqrt(2))
    return state_node


def expand(state_node):
    action = random.choice(state_node.untried_actions)
    return state_node.children[action].sample_state()


def ucb1(action_node, parent, c):
    """
    The Upper Confidence Bounds computation
    :param action_node:
    :param parent:
    :param c:
    :return:
    """
    if c == 0:  # assert that no nan values are returned for action_node.n = 0
        return action_node.q

    return action_node.q + c * np.sqrt(2 * np.log(parent.n) / action_node.n)


def best_child(state_node, c):
    """
    Returns a state node sample from the child action node with the highest
    confidence bound,

    :param state_node: The parent node
    :param c: A parameter weighting the bounds
    :return: A state node
    """
    ucb = functools.partial(ucb1, parent=state_node, c=c)
    best_action_node = _rand_max(state_node.children.values(), key=ucb)
    return best_action_node.sample_state()


def bellman_backup(node, gamma):
    while node is not None:
        node.n += 1
        if isinstance(node, StateNode):
            node.q = max([x.q for x in node.children.values()])
        elif isinstance(node, ActionNode):
            n = sum([x.n for x in node.children.values()])
            node.q = sum([(gamma * x.q + x.reward) * x.n
                          for x in node.children.values()]) / n
        node = node.parent


def _rand_max(iterable, key=None):
    if key is None:
        key = lambda x: x

    max_v = -np.inf
    max_l = []

    for item, value in zip(iterable, [key(i) for i in iterable]):
        if value == max_v:
            max_l.append(item)
        elif value > max_v:
            max_l = [item]
            max_v = value

    return random.choice(max_l)


