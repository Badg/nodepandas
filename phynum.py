import pandas as pd
import numpy as np

def IGM_interp(df, coords, guess=None):
    """Uses an approximate numerical gradient to interpolate nearby values.

    "Iterative gradient method" interpolation (totally bullshit made-up name) 
    weights points by proximity, 
    """
    # Get columns with data points
    ignorecols = ['x', 'y', 'z', 'rDist', 'rWeight']
    datcols = [col for col in df.columns if col not in ignorecols]
    fullcols = df.columns
    # First, transform to a local coordinate system with coords as [0,0,0]
    # Note that, as python is pass-by-assignment, this will not change the
    # df in the caller's namespace
    df.loc[:, 'x'] -= coords[0]
    df.loc[:, 'y'] -= coords[1]
    df.loc[:, 'z'] -= coords[2]
    # Append "rDist" to the df
    df.loc[:, 'rDist'] = (df.loc[:, 'x'] ** 2 +
        df.loc[:,'y'] ** 2 + df.loc[:, 'z'] ** 2) ** (1/2)
    # Sort by rDist and re-index
    df.sort(columns='rDist', inplace=True)
    df.set_index([list(range(len(df)))], inplace=True)
    # Handle zero rDist
    if df.ix[0, 'rDist'] == 0:
        return df.ix[0, fullcols]

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
    