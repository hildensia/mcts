from __future__ import division
import numpy as np


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
