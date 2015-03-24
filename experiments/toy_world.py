from __future__ import division
from __future__ import print_function

import random
import argparse

import numpy as np

from mcts.mcts import mcts_search
from mcts.states import toy_world_state as state
from mcts.graph import StateNode


try:
    import cPickle as pickle
except ImportError:
    import pickle
import datetime

__author__ = 'johannes'


def run_experiment(intrinsic_motivation, gamma, c, mc_n, runs, steps):
    trajectories = []
    start = np.array([50, 50])
    true_belief = True

    for _ in range(runs):
        goal = draw_goal(start, 6)
        manual = draw_goal(start, 3)
        print("Goal: {}".format(goal))
        print("Manual: {}".format(manual))

        world = state.ToyWorld([100, 100], intrinsic_motivation, goal, manual)
        belief = None
        if true_belief:
            belief = dict(zip([state.ToyWorldAction(np.array([0, 1])),
                               state.ToyWorldAction(np.array([0, -1])),
                               state.ToyWorldAction(np.array([1, 0])),
                               state.ToyWorldAction(np.array([-1, 0]))],
                              [[10, 10, 10, 10], [10, 10, 10, 10],
                               [10, 10, 10, 10], [10, 10, 10, 10]]))
        root_state = state.ToyWorldState(start, world, belief=belief)
        print(root_state.pos)
        next_state = StateNode(None, root_state, 0)
        trajectory =[]
        for _ in range(steps):
            try:
                ba = mcts_search(next_state, gamma, c=c, n=mc_n)
                print("")
                print("=" * 80)
                print("State: {}".format(next_state.state))
                print("Belief: {}".format(next_state.state.belief))
                print("Reward: {}".format(next_state.reward))
                print("N: {}".format(next_state.n))
                print("Q: {}".format(next_state.q))
                print("Action: {}".format(ba.action))
                trajectory.append(next_state.state.pos)
                if (next_state.state.pos == np.array(goal)).all():
                    break
                next_s = next_state.children[ba].sample_state(real_world=True)
                next_state = next_s
                next_state.parent = None
            except KeyboardInterrupt:
                break
        trajectories.append(trajectory)
        with open(gen_name("trajectories", "pkl"), "w") as f:
            pickle.dump(trajectories, f)
        print("=" * 80)


def draw_goal(start, dist):
    delta_x = random.randint(0, dist)
    delta_y = dist - delta_x
    return start - np.array([delta_x, delta_y])


def gen_name(name, suffix):
    datestr = datetime.datetime.strftime(datetime.datetime.now(),
        '%Y-%m-%d-%H:%M:%S')
    return name + datestr + suffix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment for UCT with '
                                                 'intrinsic motivation.')
    parser.add_argument('--intrinsic', '-i', action='store_true',
                        help='Should intrinsic motivation be used?')
    parser.add_argument('--mcsamples', '-m', type=int, default=500,
                        help='How many monte carlo runs should be made.')
    parser.add_argument('--runs', '-r', type=int, default=10,
                        help='How many runs should be made.')
    parser.add_argument('--steps', '-s', type=int, default=100,
                        help="Maximum number of steps performed.")
    parser.add_argument('--gamma', '-g', type=float, default=0.6,
                        help='The learning rate.')
    parser.add_argument('--uct_c', '-c', type=float, default=10,
                        help='The UCT parameter Cp.')

    args = parser.parse_args()
    run_experiment(intrinsic_motivation=args.intrinsic, gamma=args.gamma,
                   mc_n=args.mcsamples, runs=args.runs, steps=args.steps,
                   c=args.uct_c)


