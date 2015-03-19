__author__ = 'johannes'

import pytest
from mcts.uct import *
from mcts.uct import _rand_max
from mcts.toy_world_state import *
import numpy as np
import functools

parametrize_gamma = pytest.mark.parametrize("gamma",
                                            [.1, .2, .3, .4, .5, .6, .7, .8,
                                            .9])

parametrize_n = pytest.mark.parametrize("n", [1, 10, 23, 100, 101])


class UCBTestState(object):
    def __init__(self):
        self.actions = [0]


def test_ucb1():
    parent = StateNode(None, UCBTestState(), 0)
    an = parent.children[0]

    an.n = 1
    parent.n = 1
    assert ucb1(an, parent, 1) == 0
    ucb = functools.partial(ucb1, parent=parent, c=1)
    ucb_1 = functools.partial(ucb1, parent=parent, c=1)
    assert ucb(an) == 0
    assert ucb_1(an) == 0

    an.n = 0
    parent.n = 1
    assert np.isnan(ucb1(an, parent, 1))
    ucb = functools.partial(ucb1, parent=parent, c=1)
    assert np.isnan(ucb(an))
    assert np.isnan(ucb_1(an))

    an.n = 1
    parent.n = 0
    assert np.isnan(ucb1(an, parent, 1))
    ucb = functools.partial(ucb1, parent=parent, c=1)
    assert np.isnan(ucb(an))
    assert np.isnan(ucb_1(an))

    an.q = 1
    an.n = 1
    parent.n = 1
    assert ucb1(an, parent, 1) == 1
    ucb = functools.partial(ucb1, parent=parent, c=1)
    assert ucb(an) == 1
    assert ucb_1(an) == 1

    an.q = 19
    an.n = 0
    assert ucb1(an, parent, 0) == 19


class ComplexTestState(object):
    def __init__(self, name):
        self.actions = [ComplexTestAction('a'), ComplexTestAction('b')]
        self.name = name

    def perform(self, action):
        return ComplexTestState(action.name)

    def is_terminal(self):
        return False

    def reward(self, parent, action):
        return 0

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        return self.name == other.name


class ComplexTestAction(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        return self.name == other.name


def test_best_child():
    parent = StateNode(None, ComplexTestState('root'), 0)
    an0 = parent.children[ComplexTestAction('a')]
    an1 = parent.children[ComplexTestAction('b')]

    an0.q = 2
    an0.n = 1
    an1.q = 1
    an1.n = 1

    assert len(parent.children.values()) == 2

    assert best_child(parent, 0).state.name == 'a'


def test_rand_max():
    i = [1, 4, 5, 3]
    assert _rand_max(i) == 5

    i = [1, -5, 3, 2]
    assert _rand_max(i, key=lambda x:x**2) == -5

    parent = StateNode(None, ComplexTestState('root'), 0)
    an0 = parent.children[ComplexTestAction('a')]
    an1 = parent.children[ComplexTestAction('b')]

    an0.q = 2
    an0.n = 1
    an1.q = 1
    an1.n = 1

    ucb = functools.partial(ucb1, parent=parent, c=0)
    assert _rand_max(parent.children.values(),
                              lambda x: x.q).action.name == 'a'

    assert _rand_max(parent.children.values(),
                              ucb).action.name == 'a'


def test_untried_actions():
    s = ComplexTestState('root')
    sn = StateNode(None, s, 0)
    assert ComplexTestAction('a') in sn.untried_actions
    assert ComplexTestAction('b') in sn.untried_actions

    sn.children[ComplexTestAction('a')].n = 1
    assert ComplexTestAction('a') not in sn.untried_actions
    assert ComplexTestAction('b') in sn.untried_actions


def test_sample_state():
    s = ComplexTestState('root')
    root = StateNode(None, s, 0)

    child = root.children[ComplexTestAction('a')]
    child.sample_state()

    assert len(child.children.values()) == 1
    assert ComplexTestState('a') in child.children

    child.sample_state()
    assert len(child.children.values()) == 1
    assert ComplexTestState('a') in child.children


@pytest.fixture
def toy_world_root():
    world = ToyWorld((100, 100), False, (10, 10))
    # belief = dict(zip([ToyWorldAction(np.array([0, 1])),
    #                    ToyWorldAction(np.array([0, -1])),
    #                    ToyWorldAction(np.array([1, 0])),
    #                    ToyWorldAction(np.array([-1, 0]))],
    #                   [[10, 1, 1, 1], [1, 10, 1, 1], [1, 1, 10, 1],
    #                    [1, 1, 1, 10]]))

    state = ToyWorldState((0,0), world)

    root = StateNode(None, state, 0)

    return root, state


@parametrize_gamma
def test_single_run_uct_search(toy_world_root, gamma):
    root, state = toy_world_root
    random.seed()

    best_child = uct_search(root=root, gamma=gamma, n=1)

    states = [state for states in [action.children.values()
                                   for action in root.children.values()]
              for state in states]

    assert len(states) == 2

    assert (len(list(root.children[best_child].children.values())) == 1)

    expanded = None
    for action in root.children.values():
        if (action.action != best_child and
                    len(list(action.children.values())) == 1):
            assert expanded is None
            expanded = action

    for state in states:
        assert (np.sum(np.array(list(state.state.belief.values()))) - 1 ==
                np.sum(np.array(list(root.state.belief.values()))))
    assert root.n == 1

    for action in root.children.values():
        if action.action == expanded.action:
            assert action.q == -1.0
        else:
            assert action.q == 0.0


@parametrize_gamma
@parametrize_n
def test_n_run_uct_search(toy_world_root, gamma, n):
    root, state = toy_world_root
    random.seed()

    best_child = uct_search(root=root, gamma=gamma, n=n)

    assert root.n == n

    action_nodes, state_nodes = depth_first_search(root,
                                                   get_actions_and_states)

    for action in action_nodes:
        assert action.n == np.sum([state.n
                                   for state in action.children.values()])

    for state in state_nodes:
        assert state.n >= np.sum([action.n
                                  for action
                                  in state.children.values()]) >= state.n - 1
        # is the latter really the case?


def get_actions_and_states(node, data):
    if data is None:
        data = ([], [])

    action_nodes, state_nodes = data

    if isinstance(node, ActionNode):
        action_nodes.append(node)
    elif isinstance(node, StateNode):
        state_nodes.append(node)

    return action_nodes, state_nodes


def depth_first_search(root, fnc=None):
    data = None
    stack = [root]
    while stack:
        node = stack.pop()
        data = fnc(node, data)
        for child in node.children.values():
            stack.append(child)
    return data

