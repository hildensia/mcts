#!/usr/bin/env python2

from setuptools import setup
import mcts

setup(
    name='pymcts',
    version=mcts.__version__,
    description='Monte Carlo Tree Search in Python',
    author='Johannes Kulick',
    author_email='johannes.kulick@ipvs.uni-stuttgart.de',
    url='http://github.com/hildensia/mcts',
    packages=['mcts'],
    requires=['numpy', 'pytest']
)

