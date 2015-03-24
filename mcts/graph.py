__author__ = 'Johannes Kulick'


class Node(object):
    def __init__(self, parent):
        self.parent = parent
        self.children = {}
        self.q = 0
        self.n = 0


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


def breadth_first_search(root, fnc=None):
    data = None
    queue = [root]
    while queue:
        node = queue.pop(0)
        data = fnc(node, data)
        for child in node.children.values():
            queue.append(child)
    return data


def depth_first_search(root, fnc=None):
    data = None
    stack = [root]
    while stack:
        node = stack.pop()
        data = fnc(node, data)
        for child in node.children.values():
            stack.append(child)
    return data


def get_actions_and_states(node, data):
    if data is None:
        data = ([], [])

    action_nodes, state_nodes = data

    if isinstance(node, ActionNode):
        action_nodes.append(node)
    elif isinstance(node, StateNode):
        state_nodes.append(node)

    return action_nodes, state_nodes