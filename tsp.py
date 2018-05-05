import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#  input is list of x and y values for cities.
def solve_tsp(X, Y, max_iter=1000):

    """ solve_tsp:

        inputs
            X: array of x coords for cities
            y: array of y coords for cities
            max_iter: maximum iterations before convergence

        outputs
            ax: the axis object for the current best state and iterations number

        Methodology
            Uses a genetic algorithm to attempt to solve the traveling salesman problem
    """

