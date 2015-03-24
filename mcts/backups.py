from __future__ import division
from .graph import StateNode, ActionNode


class Bellman(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, node):
        while node is not None:
            node.n += 1
            if isinstance(node, StateNode):
                node.q = max([x.q for x in node.children.values()])
            elif isinstance(node, ActionNode):
                n = sum([x.n for x in node.children.values()])
                node.q = sum([(self.gamma * x.q + x.reward) * x.n
                              for x in node.children.values()]) / n
            node = node.parent


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