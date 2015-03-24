from __future__ import division

from mcts.states.toy_world_state import *


def test_perform():
    n = 1000

    world = ToyWorld((100, 100), False, (0, 0), np.array([100, 100]))
    belief = dict(zip([ToyWorldAction(np.array([0, 1])),
                       ToyWorldAction(np.array([0, -1])),
                       ToyWorldAction(np.array([1, 0])),
                       ToyWorldAction(np.array([-1, 0]))],
                      [[10, 1, 1, 1], [1, 10, 1, 1], [1, 1, 10, 1],
                       [1, 1, 1, 10]]))

    state = ToyWorldState((0,0), world, belief)

    outcomes = np.array([0., 0, 0, 0])
    for i in range(n):
        new_state = state.perform(state.actions[0])
        #print(new_state.belief[state.actions[0]])

        if new_state.belief[state.actions[0]][0] == 11:
            outcomes[0] += 1
        elif new_state.belief[state.actions[0]][1] == 2:
            outcomes[1] += 1
        elif new_state.belief[state.actions[0]][2] == 2:
            outcomes[2] += 1
        elif new_state.belief[state.actions[0]][3] == 2:
            outcomes[3] += 1

    print(outcomes)

    deviation = 3./np.sqrt(n)
    outcomes /= float(n)
    print(outcomes)
    expectation = np.array(belief[state.actions[0]])/\
                  sum(belief[state.actions[0]])

    assert (expectation - deviation < outcomes).all()
    assert (outcomes < expectation + deviation).all()


