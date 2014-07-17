import pandas as pd
import numpy as np
from scipy.stats import gmean
from pyDOE import lhs
import random as rand
import math
import phynum as phn
import phynum.stochastics as stoch

def mod_cuckoo(n_gens, nest_seed, nest_limits, param_limits, param_names=None, 
    keep_hist=False, fineness=1, step_adj=.9, walk="levy_flight"):
    """Modified cuckoo search!

    Starts with many nests and decreases number of nests until convergence OR
    until the minimum number of nests is reached

    n_gens: number of generations
    nest_seed: initial guessed nests
    vardef: upper, lower bounds of particle position
    nest_limits: [min, max] number of nests
    fineness: Step scale factor (greater -> smaller step size), should be > 1?

    s.A -> step size; increasing decreases step size
    s.pwr -> power of step size reduction per generation
    s.flight -> type of random walk
    s.nesd -> eggs deleted per generation
    s.constrain -> constrain to vardef?
    s.pa -> "fraction discard"

    p, F, pg, eval_hist, diversity
    p -> time history of nest position
    f -> time history of objective function value of each nest
    pg -> optimum position found
    evalhist -> number of function evaluations each generation
    diversity -> a diversity measure

    Is this maximizing or minimizing?

    Adapted from pseudocode in:
    S.Walton, O.Hassan, K.Morgan and M.R.Brown "Modified cuckoo search: A
    new gradient free optimisation algorithm" Chaos, Solitons & Fractals Vol
    44 Issue 9, Sept 2011 pp. 710-718 DOI:10.1016/j.chaos.2011.06.004
    
    Sean Walton's original MCS matlab script was used as reference, which
    is available at https://code.google.com/p/modified-cs/
    """
    # Dimensions and nests
    n_params = len(param_limits)
    n_nests = len(nest_seed)
    min_nests = 10
    # Minimum number of nests to keep every generation
    nests_to_keep = .3
    # Function to adjust step size every iteration
    adj_step = lambda x: x * step_adj
    # Number of steps to perform each levi walk
    n_levi_steps = 1

    # Check input arguments for consistency
    if n_params != len(nest_seed[1,:]):
        raise ValueError("Every row in nest_seed must have exactly one "
            "limit per parameter.")

    # If parameter names are not defined, number them
    if not param_names:
        param_names = list(range(n_params))

    # Iterant counting how many evaluations of the objective function
    itr = 0

    # Reference to objective function (currently, placeholder)
    objective = lambda x: x
    # NOTE: Need to fix this to be something legit
    # Should probably take objective as an argument

    # Get initial absolute step size pd vector for first iteration
    stepsize = pd.Series((vardef[2,:] - vardef[1,:]) / fineness)

    # Initialize the generation state variable, [<objective>, <parameters>]
    gen_state = pd.DataFrame(columns=['nest', 'objective', 'diversity'] + 
        param_names)
    hist = []

    # Initialize the state from the nest seed
    for i in range(len(nest_seed)):
        gen_state.ix[i, 0:] = nest_seed[i]
        gen_state.loc[i, 'nest'] = i

    # Calculate the objective function for the seeded parameters
    gen_state.loc[:, 'objective'] = gen_state.apply(objective, axis='index')
    # NOTE: this ^^ needs serious troubleshooting
    # Also, objective function MUST return real number

    # Calculate diversity - which is basically dist / (max dist * n_nests)
    # Deviation from mean position
    pos_dev = gen_state.ix[param_names].sub(
        gen_state.ix[param_names].mean(axis='columns'), axis='index')
    # Get the diagonal distance from all zero parameters
    pos_dist = pos_dev.mul(pos_dev).sum(axis='index').apply(np.sqrt)
    # Add diversity to state df
    gen_state.loc[:, 'diversity'] = pos_dist.sum(axis='columns') / (
        pos_dist.max() * n_nests)

    # Append the current state df to the history if desired
    if keep_hist:
        hist.append(gen_state)

    ##############################################

    # Iterate over generations
    while itr < n_gens:
        # Up the iteration number
        itr += 1

        # Sort ascending by objective
        gen_state.sort(axis='index', columns='objective', inplace=True)

        # This would be a good time to ADJUST NUMBER OF NESTS BASED ON 
        # DIVERSITY!

        # Note that int() truncates towards zero, so by adding .5 we round
        n_keep = int(n_nests * nests_to_keep + .5)
        # Slicer start for discarding eggs is the same as n_keep currently, 
        # but this allows for more flexibility going forward
        i_discard = n_keep
        # (same because indexed from zero vs raw number)

        # Deal with discarded eggs


        # perform parameter walk
        delta_params = walk







        # actual stepsize = vector stepsize / (generation number ^ power arg)
        # Scale stepsize for next generation
        stepsize = stepsize.apply(adj_step)


def empty_nests(nest, pa):
    """ Discover a fraction of the worse nests with a probability of pa.

    0 <= pa <= 1

    In the real world, if a cuckoo's egg is very similar to a host's eggs, then 
    this cuckoo's egg is less likely to be discovered, thus the fitness should 
    be related to the difference in solutions.  Therefore, it is a good idea 
    to do a random walk in a biased way with some random step sizes."""

    n = len(nest)
    comparator = [rand.random() for __nothing__ in range(n)]
    # Boolean list for discovery
    discovered = [k > pa for k in comparator]

    stepsize = rand.random() * rand.shuffle(nest)

