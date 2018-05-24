[![Build Status](https://travis-ci.org/hildensia/mcts.svg?branch=master)](https://travis-ci.org/hildensia/mcts)
[![Coverage Status](https://coveralls.io/repos/hildensia/mcts/badge.svg)](https://coveralls.io/r/hildensia/mcts)
#scikit.mcts#

Version: 0.1 (It's still alpha, don't use it for your production website!)

Website: https://github.com/hildensia/mcts

An implementation of Monte Carlo Search Trees in python.

## Setup
Requirements:
 * numpy
 * scipy
 * pytest for tests

Than plain simple `python setup.py install`. Or use `pip`: `pip install scikit.mcts`.

## Usage
Assume you have a very simple 3x3 maze. An action could be 'up', 'down', 'left' or 'right'. You start at `[0, 0]` and there is a reward at `[3, 3]`.

    class MazeAction(object):
        def __init__(self, move):
            self.move = np.asarray(move)
        
        def __eq__(self, other):
            return all(self.move == other.move)
            
        def __hash__(self):
            return 10*self.move[0] + self.move[1]
    
    class MazeState(object):
        def __init__(self, pos):
            self.pos = np.asarray(pos)
            self.actions = [MazeAction([1, 0]),
                            MazeAction([0, 1]),
                            MazeAction([-1, 0]),
                            MazeAction([0, -1])]
        
        def perform(self, action):
            pos = self.pos + action.move
            pos = np.clip(pos, 0, 2)
            return MazeState(pos)
            
        def reward(self, parent, action):
            if all(self.pos == np.array([2, 2])):
                return 10
            else:
                return -1
                
        def is_terminal(self):
            return False
                
        def __eq__(self, other):
            return all(self.pos == other.pos)
            
        def __hash__(self):
            return 10 * self.pos[0] + self.pos[1]
            
This would be a plain simple implementation. Now let's run MCTS on top:

    mcts = MCTS(tree_policy=UCB1(c=1.41), 
                default_policy=immediate_reward,
                backup=monte_carlo)
    
    root = StateNode(parent=None, state=MazeState([0, 0]))
    best_action = mcts(root)


## Licence
See LICENCE

## Authors
Johannes Kulick
            
            
