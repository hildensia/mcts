from __future__ import print_function

import functools
import random

from . import backups
from . import tree_policies
from . import default_policies
from . import utils


def uct(gamma, c=1.41):
    return functools.partial(mcts, tree_policy=tree_policies.UCB1(c),
                             default_policy=default_policies.immediate_reward,
                             backup=backups.Bellman(gamma))


def mcts(root, tree_policy, default_policy, backup, n=1500):
    for _ in range(n):
        node = _get_next_node(root, tree_policy)
        node.reward = default_policy(node)
        backup(node)

    return utils.rand_max(root.children.values(), key=lambda x: x.q).action


def _expand(state_node):
    action = random.choice(state_node.untried_actions)
    return state_node.children[action].sample_state()


def _best_child(state_node, tree_policy):
    best_action_node = utils.rand_max(state_node.children.values(),
                                      key=tree_policy)
    return best_action_node.sample_state()


def _get_next_node(state_node, tree_policy):
    while not state_node.state.is_terminal():
        if state_node.untried_actions:
            return _expand(state_node)
        else:
            state_node = _best_child(state_node, tree_policy)
    return state_node
