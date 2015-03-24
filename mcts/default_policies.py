def immediate_reward(state_node):
    return state_node.state.reward(state_node.parent.parent.state,
                                   state_node.parent.action)
