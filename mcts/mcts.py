from __future__ import print_function

import functools
import logging
import random

import backups
import tree_policies
import utils


def mcts_search(root, gamma, n=1500, c=1.41):
    logger = logging.getLogger('uct')
    for i in range(n):
        print('.', end='')
        node = tree_policy(root, c)
        #default_policy(node)
        backups.bellman_backup(node, gamma)

    logger.debug(dict([(action.action, action.q)
                       for action in root.children.values()]))
    return best_child(root, 0).parent.action


def expand(state_node):
    action = random.choice(state_node.untried_actions)
    return state_node.children[action].sample_state()


def best_child(state_node, c):
    """
    Returns a state node sample from the child action node with the highest
    confidence bound,

    :param state_node: The parent node
    :param c: A parameter weighting the bounds
    :return: A state node
    """
    ucb = functools.partial(tree_policies.ucb1, parent=state_node, c=c)
    best_action_node = utils.rand_max(state_node.children.values(), key=ucb)
    return best_action_node.sample_state()


def tree_policy(state_node, c):
    while not state_node.state.is_terminal():
        if state_node.untried_actions:
            return expand(state_node)
        else:
            state_node = best_child(state_node, c)
    return state_node