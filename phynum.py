import pandas as pd
import numpy as np
from scipy.stats import gmean
from pyDOE import lhs
import random as rand

def IGM_interp(df, coords, coordnames=['x','y','z'], ignorecols=None, guess=None):
    """Uses an approximate numerical gradient to interpolate nearby values.

    "Iterative gradient method" interpolation (totally bullshit made-up name) 
    weights points by proximity, 
    """
    # Expediently get the number of rows
    numpoints = len(df.index)
    # Immediately reindex
    df.set_index([list(range(numpoints))], inplace=True)
    # Get the column names
    colnames = df.columns
    # Assign coordinate column names to representative variables
    if len(coordnames) != 3 or \
        [col for col in coordnames if col not in colnames]: 
        raise ValueError("coordnames must have 3 elements and all must be "
            "present in dataframe")
    x, y, z = coordnames
    # Get columns with data points
    if not ignorecols:
        ignorecols=[]
    ignorecols.extend(['rDist', 'rWeight'])
    ignorecols.extend(coordnames)
    datcols = [col for col in df.columns if col not in ignorecols]

    # Transform to a local coordinate system with coords as [0,0,0]
    # Note that, as python is pass-by-assignment, this will not change the
    # df in the caller's namespace
    df.loc[:, x] -= coords[0]
    df.loc[:, y] -= coords[1]
    df.loc[:, z] -= coords[2]

    # Generate skew-symmetric matrices for each of the fields for deltas
    zeros = pd.DataFrame(index=range(numpoints), 
        columns=range(numpoints)).fillna(0)
    deltas = {}
    for col in colnames:
        deltas[col] = zeros.add(df.ix[:,col], axis='index', level='columns')\
            .sub(df.ix[:,col], axis='columns', level='index')

    # Create the multipliers dict used to weight the partial derivatives
    alphas = {}
    denom = {}
    partials = {}
    partials_norm = {}
    variance = {}
    # This should be reworked to use more loopage
    for col in datcols:
        alphas[col] = {x:1, y:1, z:1}

        # Calculate the denominator of the weights expression
        denom[col] = (alphas[col][x] * deltas[x]).add(
            alphas[col][y] * deltas[y]).add(
            alphas[col][z] * deltas[z])

        # Calculate the pdVal/pdCoords. These are symmetric (not skew)
        # Also calculate them normalized as percentages.
        partials[col] = {
            x: (alphas[col][x] * deltas[col]).div(denom[col]),
            y: (alphas[col][y] * deltas[col]).div(denom[col]), 
            z: (alphas[col][z] * deltas[col]).div(denom[col])
            }
        partials_norm[col] = {
            x: partials[col][x].div(df.loc[:, col], level='columns') * 100,
            y: partials[col][y].div(df.loc[:, col], level='columns') * 100, 
            z: partials[col][z].div(df.loc[:, col], level='columns') * 100
            }

        # Compute the variance of the normalized partials
        variance[col] = np.prod([
            partials_norm[col][x].var(ddof=0, axis='index').mul( 
            partials_norm[col][y].var(ddof=0, axis='index')).mul( 
            partials_norm[col][z].var(ddof=0, axis='index'))
            ])

    # Sum the partials
    # summation = {}
    # for col in datcols:
    #     summation[col] = pdX[col].mul(deltas[x]) + \
    #     pdY[col].mul(deltas[y]) + \
    #     pdZ[col].mul(deltas[z])

    print("##########################################################")
    print(partials_norm)
    print(variance)

    # Append "rDist" to the df
    df.loc[:, 'rDist'] = (df.loc[:, 'x'] ** 2 +
        df.loc[:,'y'] ** 2 + df.loc[:, 'z'] ** 2) ** (1/2)
    # Sort by rDist and re-index
    df.sort(columns='rDist', inplace=True)
    # Handle zero rDist
    if df.ix[0, 'rDist'] == 0:
        return df.ix[0, colnames]

    # Use the mean of the two closest points as a guess if one is not passed
    if not guess:
        guess = df.ix[:, datcols].mean()

    # Get initial partial derivatives from guess. Don't forget that coords
    # have been adjusted to center around 0, 0, 0, so no subtraction is needed

    # Now iterate!
    for ii in range(len(df)):
        # Slice the distances along the go
        df_dists = df.ix[:ii, 'rDist']
        # Get the delta values, delta position, rotate and replace 0 dists 
        # with NaN to indicate missing partials
        dA = guess.ix[datcols] - df.ix[:ii, datcols] 
        dS = pd.DataFrame([df.ix[:ii, 'x'], df.ix[:ii, 'y'], 
            df.ix[:ii, 'z']]).transpose().replace(0, np.nan) * -1
        # Get weights based on radial distance, x, y, and z
        weights = pd.DataFrame([get_cubic_weight(df_dists), 
            get_cubic_weight(dS.loc[:, 'x']), 
            get_cubic_weight(dS.loc[:, 'y']), 
            get_cubic_weight(dS.loc[:, 'z'])]).transpose()
        # Get the constants for dissection of deltaA/deltaS into partials.
        # Note that the alphas are the only thing controlling what amount of
        # value change is attributed to the x, y, and z directions. Currently
        # it is equal to the x, y, and z changes divided by the radial dists.
        # It needs to have columns of 'x', 'y', and 'z'.
        alphas = dS.div(weights.loc[:, 'rDist'], axis='index').abs()

        # Get the denominators for the dissection equation
        denoms = alphas.mul(dS).sum(axis='columns')

        # Get the partial derivative estimates from each datapoint
        dAdX = dA.mul(alphas.loc[:,'x'], axis='index', level='columns').div(
            denoms, axis='index', level='columns')
        dAdY = dA.mul(alphas.loc[:,'y'], axis='index', level='columns').div(
            denoms, axis='index', level='columns')
        dAdZ = dA.mul(alphas.loc[:,'z'], axis='index', level='columns').div(
            denoms, axis='index', level='columns')

        # Combine the individual datapoint estimates into single values
        # Is this actually a good thing to do?  You're decreasing data: 
        # you're starting with multiple estimates of partial derivatives,
        # which should give you an idea of spatial change.
        partials = pd.DataFrame([
            dAdX.mul(weights.loc[:, 'x'], axis='index', level='columns').sum(),
            dAdY.mul(weights.loc[:, 'y'], axis='index', level='columns').sum(),
            dAdZ.mul(weights.loc[:, 'z'], axis='index', level='columns').sum()
            ], index=['x', 'y', 'z'])

        # Use the computed partials to update the guess, integrating from the
        # closest point
        deltas = pd.DataFrame({
            'x': 
            partials.loc['x', datcols].mul(df.ix[0, 'x'], level='columns'),
            'y':
            partials.loc['y', datcols].mul(df.ix[0, 'y'], level='columns'),
            'z':
            partials.loc['z', datcols].mul(df.ix[0, 'z'], level='columns')})\
            .transpose()

        guess = df.ix[0,datcols] - deltas.sum(axis='index')
    return guess

def MCS(n_gens, nest_seed, param_limits=None, param_names=None):
    """Modified cuckoo search!

    Adapted from Sean Walton's MCS matlab script.  Relevant paper citation:
    S.Walton, O.Hassan, K.Morgan and M.R.Brown "Modified cuckoo search: A
    new gradient free optimisation algorithm" Chaos, Solitons & Fractals Vol
    44 Issue 9, Sept 2011 pp. 710-718 DOI:10.1016/j.chaos.2011.06.004

    n_gens: number of generations
    nest_seed: initial guessed nests
    vardef: upper, lower bounds of particle position

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
    """
    # Dimensions and nests
    n_params = len(nest_seed[1,:])
    n_nests = len(nest_seed)
    min_nests = 10

    # Check input arguments for consistency
    if param_limits and n_params != len(param_limits):
        raise ValueError("Every row in nest_seed must have exactly one "
            "limit per parameter.")

    # If parameter names are not defined, number them
    if not param_names:
        param_names = list(range(n_params))

    # Iterant counting how many evaluations of the objective function
    ii = 0

    # Reference to objective function (currently, placeholder)
    objective = lambda x: x
    # NOTE: Need to fix this to be something legit
    # Should probably take objective as an argument

    # Step scale factor (greater -> smaller step size)
    fineness = 1
    stepsize = (vardef[1,:] - vardef[2,:]) / fineness

    # timeStart = tic;


    # Initialize the generation state variable, [<objective>, <parameters>]
    gen_state = pd.DataFrame(columns=['nest', 'objective', 'diversity'] + 
        param_names)
    diversity = []
    eval_hist = []

    # Initialize the state from the nest seed
    for i in range(len(nest_seed)):
        gen_state.ix[i, 0:] = nest_seed[i]
        gen_state.loc[i, 'nest'] = i

    # Calculate the objective function for the seeded parameters
    gen_state.loc[:,'objective'] = gen_state.apply(objective, axis='index')
    # NOTE: this ^^ needs serious troubleshooting


    ##############################################

    
    pos_current = nest_seed
    # Current objective: <number of nests>
    # Changereq: This should use numpy arrays for better performance
    obj_current = []
    # Calculate fitness for initial nests
    for i in range(n_nests):
        obj_temp = objective(pos_current[i])
        if obj_temp.imag:
            obj_current.append(np.inf)
            # Changereq: np.NaN
        else:
            obj_current.append(obj_temp)
    # THIS is why the iteration number history is needed -- it's saving the
    # histories and you need to be able to distinguish the nests.
    # Should probably allow the option to do either.

    positions.append(pos_current)
    objectives.append(obj_current)

    # Calculate diversity
    pos_mean = pos_current.mean(axis='columns')
    pos_dist = np.square(pos_current.sub(pos_mean, axis='index'))\
        .sum(axis='index')
    length_diag = max(pos_dist.append(0))
    diversity.append(pos_dist.sum(axis='columns') / (length_diag * n_nests))
    eval_hist.append(n_iter)

    # pa = S.pa; (fraction to discard)
    # ptop = 1-pa;
    # G = 1;

    # Iterate over generations
    while n_iter < n_gens:
        # Up the iteration number
        n_iter += 1




def lhs_scaled(var_min, var_max, n):
    """ Latin Hypercube Sampling of variables defined by vardef
    vardef(2, len(N)) <max value, min value>
    N -> number of samples to generate
    LHC.m
    """

    # Get number of variables and their scaling factor
    k = len(var_min)
    spread = var_max - var_min

    # Get linear hypercube sample and scale it, then return
    return var_min + spread * lhs(k, n, criterion='maximin', iterations=100)

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


def linterp_1D(series1, series2, colname, coord, ignore=None):
    """Simple 1d linear interpolator between 2 pandas series.
    """
    if ignore:
        ignore.append(colname)
    else:
        ignore = [colname]    

    index1 = series1.index
    index2 = series2.index

    datcols = [col for col in index1 if col not in ignore and col in index2]

    x1 = series1.loc[colname]
    x2 = series2.loc[colname]

    if x1 == x2:
        # HOW TO HANDLE THIS?!?!?!
        #########################################################
        #########################################################
        pass

    return series1.loc[datcols] + (coord - x1) * (series2.loc[datcols] - 
        series1.loc[datcols]) / (x2 - x1)

def get_cubic_weight(pd_ser):
    """Assigns a normalized (w1+w2=1) weight to pd_ser, decaying with r^3.
    """
    closest = min(pd_ser)
    weights = (closest / pd_ser) ** 3
    return (weights / weights.sum()).fillna(0)

def pd_dist(pd_ser1, pd_ser2 = pd.Series({'x': 0, 'y': 0, 'z': 0}), 
    coords=['x', 'y', 'z']):
    """ Gets a distance between 2 points in cartesian space, or to the origin.
    """
    # Predeclare total to zero
    total = 0
    for coord in coords:
        if coord in pd_ser1.index and coord in pd_ser2.index:
            total += (pd_ser2.loc[coord] - pd_ser1.loc[coord]) ** 2
        else:
            raise KeyError("Key " + coord + " missing from series.")
    return total ** (1/2)
        
def get_dist(coords1, coords2):
    """ Gets the scalar distance from coords1 to coords2.
    
    +"coords1" and "coords2" are lists.  If one is longer than the other, 
    execution halts at the end of the shortest list.
    """
    deltas = []
    summe = 0
    
    for coord1, coord2 in zip(coords1, coords2):
        deltas.append(coord2-coord1)
    
    for delta in deltas:
        summe += delta**2
    
    return (summe ** .5)

def get_deltas(iterable):
    """ Calculates the deltas between adjacent elements in an iterable.
    
    + Iterable is an iterable of length minimum n = 2
    
    Returns an iterable of length n - 1.
    """
    # Initialize the offset variable with an "empty" element
    offset = [0]
    offset.extend(iterable)
    # Create a list of spread-normalized deltas (first will be zero)
    deltas = [(it - off) for off, it in zip(offset, iterable)]
    # Remove first element so it doesn't screw up calculations
    del deltas[0]
    
    return deltas

def knn_coords(df, k, coords, scale_length=None):
    """ Searches a pandas <df> for the <k> nearest neighbours to <coords>.
    
    Arguments
    =========
    + <df> : pandas dataframe to search
    + <k> : integer number of neighbors to find
    + <coords> : tuple or list of coordinates to search
    + <scale_length> : optional approximate side length of each coordinate 
    cell.  Used to preselect search bucket for faster search and highly 
    recommended for any large <df>.
    
    Returns
    =======
    + <nameless> : pandas dataframe
    """    
    # Extract coords
    x = coords[0]
    y = coords[1]
    z = coords[2]
    
    # Create the search bucket
    bucket = pd.DataFrame()
    bucketslop = 2
    
    # If scale length is defined, shrink the search bucket
    if scale_length:
        # Conservatively select a "cube" of data with side length bucket_leg
        # centered at coords
        bucket_leg = k * scale_length / 2
        # If the bucket is smaller than bucketslop*k (cannot be k, since this 
        # is searching manhattan distance and we want euclidian distance), 
        # select a bucket and then (just in case) scale the bucket_leg
        while len(bucket) < k * bucketslop:
            bucket = df[(df.x - x > -bucket_leg) & (df.x - x < bucket_leg) &
                        (df.y - y > -bucket_leg) & (df.y - y < bucket_leg) &
                        (df.z - z > -bucket_leg) & (df.z - z < bucket_leg)]
            # Double the bucket leg length in case this is too small
            bucket_leg *= 2
    # If scale length isn't defined, use the whole df
    else:
        bucket = df
        
    # Progress report: we have a search bucket.  Now let's search it.
    # Examine each row in the bucket and calculate the distance to target, 
    # creating a zip of [(<distance>, <index>)]
    dists = ((bucket.loc[:,'x'] - x)**2 + (bucket.loc[:,'y'] - y)**2 + 
             (bucket.loc[:,'z'] - z)**2)**(1/2)
    dists = zip(dists, range(len(dists)))
    
    # Now sort dists ascendingly and select the k indices
    dists = sorted(dists, key=lambda item: item[0])
    dists = dists[:k]
    sel = [it[1] for it in dists]
    
    # Return the corresponding first k elements from bucket
    return bucket.iloc[sel]
    