import math
import numpy as np
from itertools import tee
import matplotlib.pyplot as plt
import seaborn as sns

import time

from IPython import display


def distance(points):
    """ distance

        input
            point: tuple of tuples ((x1, y1), (x2, y2))

        output
            distance: the distance between the two points
    """
    return math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)


def pairwise(iterable):

    """ pairwise

        [a, b, c, d] => (a,b)->(b,c)->(c,d)
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_distance(cities, x, y):

    """ get_distance:

        inputs
            cities: list of indices for order of cities for population member

            x: array of x coords for cities
            y: array of y coords for ciites

        outputs
            total distance for cities array
    """

    return sum([distance(i) for i in pairwise(zip(x[cities], y[cities]))])


def solve_tsp(X, Y, max_iter=1000, pop_size=100):

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

    num_cities = X.size

    #  create the initial populaiton
    pop = np.random.rand(pop_size, num_cities).argsort(axis=-1)

    #  initialize distance array
    distances = np.apply_along_axis(get_distance, 1, pop, X, Y)

    #  initial best
    best_idx = np.argmin(distances)
    best_path = pop[best_idx, :]
    best = distances[best_idx]
    all_best = [np.amax(distances), best]
    best_gens = [0, 1]

    #  build initial plot
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    fig.suptitle("GA TSP Solver\nGen: 0 Distance: " + str(best))
    ax1.set_title("Current Best Path")
    ax2.set_title("Current Distance")

    ax1.plot(X[best_path], Y[best_path], 'r')
    ax1.scatter(X[best_path], Y[best_path])

    ax2.set_xlim(0, 100)
    ax2.set_ylim(np.amax(distances), 0)


    for gen in range(0 + 1, max_iter + 1):
        new_pop = pop

        #  tournament selection and mutation
        pop_idx = np.random.permutation(pop_size)
        for idxs in [pop_idx[i:i+4] for i in range(0, pop_size, 4)]:

            #  index of in pop of winner of tournament
            win_idx = idxs[np.argmin(distances[idxs])]
            winner = pop[win_idx, :]

            #  two random pivot points
            pivots = sorted(np.random.choice(num_cities, 2))
            p1, p2 = pivots[0], pivots[1]

            for i, idx in enumerate(idxs):

                #  set winner
                if i == 0:
                    new_pop[idx, :] = winner

                #  flip
                if i == 1:
                    temp = np.copy(winner)
                    temp[p1:p2+1] = winner[p1:p2+1][::-1]
                    new_pop[idx, :] = temp
                #  swap
                if i == 2:
                    temp = np.copy(winner)
                    temp[[p1, p2]] = winner[[p2, p1]]
                    new_pop[idx, :] = temp

                #  slide
                if i == 3:
                    temp = np.copy(winner)
                    temp[p1:p2+1] = np.roll(temp[p1:p2+1], -1)
                    new_pop[idx, :] = temp

        #  update population with update
        pop = new_pop

        #  fitness
        distances = np.apply_along_axis(get_distance, 1, pop, X, Y)

        cur_idx = np.argmin(distances)
        cur_best = distances[cur_idx]
        cur_path = pop[cur_idx, :]

        #  update new best
        if cur_best < best:
            ax1.clear()
            best_path = cur_path
            best = cur_best
            best_idx = cur_idx
            all_best.append(best)
            best_gens.append(gen)

            ax1.plot(X[best_path], Y[best_path], '-r')
            ax1.scatter(X[best_path], Y[best_path])

            ax2.plot(best_gens, all_best, '-r')

            fig.suptitle("GA TSP Solver\nGen: " + str(gen) + " Distance: " + str(best))

        if gen == max_iter:

            best_gens.append(gen)
            all_best.append(best)
            ax2.plot(best_gens, all_best, '-r')

            fig.suptitle("GA TSP Solver\nGen: " + str(gen) + " Distance: " + str(best))

            display.clear_output(wait=True)
            display.display(plt.gcf())
            time.sleep(0.05)
