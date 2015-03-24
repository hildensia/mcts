from __future__ import division
from .graph import StateNode, ActionNode

__author__ = 'Johannes Kulick'


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