""" New inversion finding routine. The old routine is too glitchy and reading the
code is annoying since it's not as well modularized as I like.

The way that this code should work is thus:
1. Check data. All we need is a temperature and height profile. No nans.
2. Identify layers based on gradient. Use iso params to make all layers 1 or -1.
3. Iterate:
    Identify top and bottom of layers by sign changes.
    Make an array with a row for each layer.
    If any inversion layers are too thin, set them to be non-inversion.
    If any inversion layers sandwich a thin noninversion layer, set the non-inversion layer to
    be an inversion layer.

Optional arguments:
'temperature': name of temperature column in data
'height': name of height column in data
'pressure': name of pressure column in data
'iso_base': if True, consider isothermal layers bordering an inversion layer from below to be part of the inversion layer. Default False.
'iso_adjacent': if True, consider isothermal layers adjacent to an inversion layer to be part of the inversion layer. Default False.
'iso_within': if True, consider isothermal layers contained within an inversion layer to be part of the inversion layer. Default False.
'iso_alone': if True, consider all isothermal layers to be inversion layers. Default False.
'na_tol': Fraction of data points allowed to be missing. Default 0.1.
'min_inv_depth': Inversion layers less than min_inv_depth are dropped. Default 20 m.
'max_lapse_rate': If any layers have a stronger gradient than this, the profile is considered invalid.
Default is 0.4 degrees per meter.


To Do List
Update clean_data function so that if height is missing but pressure and temperature are not, height can be
estimated from the geopotential height relationship.

Add a function that will process the layer_df to make a dataframe with inversion values for at least a
few preset definitions.
 
"""

import warnings
warnings.filterwarnings('ignore', 'invalid value encountered in greater', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in true_divide', RuntimeWarning)
warnings.filterwarnings('ignore', 'divide by zero encountered in true_divide', RuntimeWarning)

def default_params():
    """Default set of parameters. Parameters let the other functions
    know what column names are as well as set options for the classification
    of temperature inversions.

    TODO: Add units, at least in description of default_params
    """
    
    return {'temperature': 'temperature',
           'height': 'height',
           'pressure': 'pressure',
           'potential_temperature': 'potential_temperature',
           'iso_base': False,
           'iso_above': False,
           'iso_within': True,
           'iso_alone': False,
           'na_tol': 0.1,
           'max_embed_depth': 100,
           'min_inv_depth': 20,
           'max_lapse_rate': 0.4
           }

def check_data(data, params):
    """Checks temperature and height data.

    Input: DataFrame with 
    Requires
    - that the height vectors are numerical
    - that the number of missing data is less than the fraction specified by
      the parameter na_tol
    - that the height vector is monotonic.
    If the height vector is nonmonotonic, and there are no points where height decreases,
    then rows with height equal to the row before are dropped."""

    import numpy as np
    import numpy.testing as npt

    temperature = params['temperature']
    height = params['height']
    pressure = params['pressure']
    
    # make sure vectors for temperature, pressure, and height are present
    if ~np.all([x in data.keys() for x in (temperature, height, pressure)]):
        params['qc'] = False
        params['qc_err'] = 'temperature and height vectors missing or names specified incorrectly'
        return data, params

    n0 = len(data)
    data = data.dropna(axis=0, how='any', subset=[temperature, height, pressure]).copy()
    data.reset_index(inplace=True, drop=True)
    
    n1 = len(data)
    if n0 - n1 > n0*params['na_tol']:
        params['qc'] = False
        params['qc_err'] = 'too many missing values'
        return data, params

    # make sure vectors for temperature, pressure, and height are numerical
    real_count = data.applymap(np.isreal).sum()
    if ~np.all([real_count[temperature] == n1,
               real_count[pressure] == n1,
               real_count[height] == n1]):
        params['qc'] = False
        params['qc_err'] = 'non-numeric'
        return data, params
    
    # check that the height vector is monotonic
    if np.sum(np.diff(data[height].values) == 0) > 0:
        params['qc'] = True
        params['qc_err'] = 'Dropped repeated heights'
        drop_rows = np.concatenate(([False], data[height].values[1:] == data[height].values[:-1]))
        data.loc[drop_rows,:].reset_index(inplace=True, drop=True)

    if np.sum(np.diff(data[height].values) < 0) > 0:
        params['qc'] = False
        params['qc_err'] = 'Decreasing height'
        return data, params

    
    dtdz = np.diff(data[temperature].values)/np.diff(data[height].values)   
    # check whether the max lapse rate is exceeded
    if np.any(dtdz > params['max_lapse_rate']):
        params['qc'] = False
        params['qc_err'] = 'Exceeded max lapse rate'
        return data, params

    # return either a quality-controlled data frame and a passing qc flag
    # or the original data frame, failing flag, and an error message
    
    params['qc'] = True
    params['qc_err'] = 'Passed QC'
    return data, params

def update_sign_vector(data, params):
    """The sign vector is used to track which layers are counted
    as inversion layers. The first time this is called, a sign vector
    is created. Subsequent calls are to keep the sign vector in line
    with isothermal handling parameters.

    If sgn[idx] = 1, that means that the layer between z[idx-1] and z[idx] belongs
    to an inversion layer. 

    params:
    'iso_below': if True, consider isothermal layers bordering an inversion layer
     from below to be part of the inversion layer. Default False.
    'iso_above': if True, consider isothermal layers bordering an inversion layer
     from above to be part of the inversion layer. Default False.
    'iso_alone': if True, consider all isothermal layers to be inversion layers.
    Default False.

    Potential parameter to add later if desired: 'iso_within': if True, consider
    isothermal layers embedded within an inversion to be part of an inversion layer.
    Choosing iso_base or iso_above already does this, but there may be a time when
    embedded isothermal layers but not bordering layers are wanted.
    
    Returns:
    DataFrame: data with added 'sign' column indicating 1 for inversion and -1 for noninversion.
    Dict: parameter dictionary with added 'inv_present' value
    """

    import numpy as np
    
    if 'sign' not in data.keys():
        # When first called, create a sign vector.
        dt = np.diff(data[params['temperature']].values)
        data = data.assign(sign=np.concatenate(([-1], np.sign(dt))))

    sgn = data['sign'].values
    if np.all(sgn < 0):
        params['inv_present'] = False
        return data, params

    if params['iso_alone']:
        data.loc[sgn == 0, 'sign'] = 1
        params['inv_present'] = True
        return data, params
        
    if params['iso_base']:
        iso_idx = (sgn[:-1] == 0) & (sgn[1:] == 1)
        while np.any(iso_idx):
            sgn[:-1][iso_idx] = 1
            iso_idx = (sgn[:-1] == 0) & (sgn[1:] == 1)

    if params['iso_above']:
        iso_idx = (sgn[:-1] == 1) & (sgn[1:] == 0)
        while np.any(iso_idx):
            sgn[1:][iso_idx] = 1
            iso_idx = (sgn[:-1] == 1) & (sgn[1:] == 0)
        data.loc[:,'sign'] = sgn

    if not params['iso_alone']:
        data.loc[sgn==0, 'sign'] = -1

    params['inv_present'] = True
    return data, params

def build_layer_df(data, params):
    """Build a dataframe summarizing layers based on the current sign
    vector. Does not perform any merge operations.

    Inputs:
    data: Pandas dataframe returned from update_sign_vector
    params: Dictionary with keyword arguments
    
    Returns:
    Dataframe with layer summary.
    """
    
    import numpy as np
    import pandas as pd

    layer_df = pd.DataFrame(columns=['date','idx_base','idx_top', 'height_base', 'height_top', 'temperature_base',
                                     'temperature_top', 'pressure_base', 'pressure_top', 'layer_depth',
                                     'layer_strength', 'max_dzdt', 'qc', 'qc_err'],
                            dtype=float,
                            index=[0])

    layer_df['date'] = data.date.values[0]
    layer_df['qc'] = params['qc']
    layer_df['qc_err'] = params['qc_err']
    
    if not params['inv_present']:
        return layer_df

    if not params['qc']:
        return layer_df
    
    sgn = data['sign'].values
    t = data[params['temperature']].values
    z = data[params['height']].values
    p = data[params['pressure']].values
    
    idx_list = np.atleast_1d(np.argwhere(sgn[1:] != sgn[:-1]).squeeze())

    # Add last index if the length of idx_list is odd - that would
    # indicate an inversion layer extending past the last observation
    # level.
    if len(idx_list) % 2 == 1:
        idx_list = np.concatenate((idx_list, [len(sgn)-1]))
        layer_df['qc_err'] = 'Inversion exists at top of profile.'

    # Index arrays for base and top of layers
    idxb = idx_list[:-1]
    idxt = idx_list[1:]

    # Find max gradient in each layer
    dzdt=np.diff(t)/np.diff(z)
    max_dzdt = np.array([np.max(dzdt[i0:i1]) for i0, i1 in zip(idxb, idxt)])

    layer_df = pd.DataFrame({'idx_base': idxb,
                      'idx_top': idxt,
                      'height_base': z[idxb],
                      'height_top': z[idxt],
                      'temperature_base': t[idxb],
                      'temperature_top': t[idxt],
                      'pressure_base': p[idxb],
                      'pressure_top': p[idxt],
                      'layer_depth': z[idxt] - z[idxb],
                      'layer_strength': t[idxt] - t[idxb],
                      'max_dzdt': max_dzdt})
    
    return layer_df
    
    
def merge_layers(data, params):
    """Applies the criteria in params to specify whether a layer should
    be included as an inversion layer.

    layer_array is the array produced by layer_array
    sgn is the sign vector produced by update_sign_vector
    
    Params:
    'max_embed_depth': Embedded layers thinner than this will be merged
    'min_inv_depth': Inversion layers thinner than this will be dropped

    Returns:
    layer_array
    updated sgn vector
    """
    import numpy as np

    
    layer_df = build_layer_df(data, params)

    if not params['inv_present']:
        return layer_df

    if not params['qc']:
        return layer_df


    # Drop inversion layers thinner than the min inversion depth
    changed = False
    for x in np.argwhere((layer_df['layer_strength'] > 0) &
                         (layer_df['layer_depth'] < params['min_inv_depth'])):
        idxb = int(layer_df.loc[x, 'idx_base']) + 1
        idxt = int(layer_df.loc[x, 'idx_top'])
        data.loc[idxb:idxt, 'sign'] = -1
        changed = True

    if changed:
        data, params = update_sign_vector(data, params)
        layer_df = build_layer_df(data, params)
        if not params['inv_present']:
            return layer_df

    # Merge embedded noninversion layers
    changed = False
    for x in np.argwhere((layer_df['layer_strength'] < 0) &
                         (layer_df['layer_depth'] < params['max_embed_depth'])):
        
        if (x > 0) & (x < len(layer_df)-1):
            if np.all([layer_df.loc[x-1, 'layer_strength'].values > 0,
                       layer_df.loc[x+1, 'layer_strength'].values > 0]):
                idxb = int(layer_df.loc[x, 'idx_base']) + 1
                idxt = int(layer_df.loc[x, 'idx_top'])
                data.loc[idxb:idxt,'sign'] = 1
                changed = True
    if changed:
        data, params = update_sign_vector(data, params)
        layer_df = build_layer_df(data, params)
        if not params['inv_present']:
            return layer_df

    layer_df = layer_df[layer_df.layer_strength > 0].copy()
    layer_df.index = np.arange(len(layer_df)) + 1
    layer_df['date'] = data.date.values[0]
    return layer_df
    
def find_temperature_inversions(data, params):
    """Identify temperature inversions in a dataframe.

    data: Dataframe containing height and temperature data for one sounding.
    
    Params is a dictionary that contains parameters for the inversion 
    detection algorithm.
    
    Returns a dataframe containing a line for each inversion in the profile.
    The columns of the dataframe are 
    index of base, index of top, height of base, height of top,
    temperature of base, temperature of top, layer depth, layer strength, 
    and max gradient within inversion."""
    
    ### Could probably make use of defining a class so I wouldn't pass data and params over and over.
    data.reset_index(inplace=True)
    data, params = check_data(data, params)
    data, params = update_sign_vector(data, params)
    layer_df = merge_layers(data, params)
    layer_df.index.names = ['inv_number']
    return layer_df

def kahl_inversions(data, params):
    """Given a dataframe "data" with possibly many radiosonde profiles,
    find the lowest temperature inversion in each following Kahl 1990.
    """
    
    inv_df = data.groupby('date').apply(find_temperature_inversions, params)
    inv_df.drop('date', axis=1, inplace=True)
    inv_df.reset_index(inplace=True)
    inv_df['dT'] = inv_df['temperature_top'] - inv_df['temperature_base']
    inv_df['bh'] = inv_df['height_base']
    inv_df['dZ'] = inv_df['height_top'] - inv_df['height_base']

    inv_df = inv_df[inv_df.inv_number <= 1].copy()
    inv_df.reset_index(inplace=True, drop=True)
    inv_df.index = inv_df.date
    inv_df.drop('date', inplace=True, axis=1)
    return inv_df.loc[:,['dT', 'bh', 'dZ']]

def tmax(data, params):
    """Definition of temperature inversion applied in the IGRA2 Derived
    data set. Inversion top is defined as the temperature maximum, inversion
    base is always the surface, inversion height is the height of the temperature
    maximum, and inversion pressure is the pressure at the temperature maximum."""
    import numpy as np
    def apply_tmax(data, params):
        """Locates the line in data with the maximum temperature and the line
        with the maximum pressure, and differences them."""
           
        data_tmax = data.loc[data[params['temperature']].idxmax(),:]
        data_pmax = data.loc[data[params['pressure']].idxmax(),:]
        data_diff = data_tmax - data_pmax
        data_diff[params['pressure']] = data_tmax[params['pressure']]
        if data_diff[params['temperature']] < 0.1:
            data_diff[params['temperature']] = np.nan
            data_diff[params['pressure']] = np.nan
            data_diff[params['height']] = np.nan
        return data_diff

    inv_df = data.groupby('date').apply(apply_tmax, params).loc[:, [params['pressure'],
                                                                    params['height'],
                                                                    params['temperature']]]
    inv_df.rename({params['pressure']: 'inv_pressure',
                   params['height']: 'inv_height',
                   params['temperature']: 'inv_strength'}, axis=1)
    return inv_df

def tmax_hybrid(data):
    # same as tmax, but starting from the inversion base determined by
    # kahl1990
    inv_df = data.groupby('date').apply(find_temperature_inversions, params)
    inv_df.drop('date', axis=1, inplace=True)
    inv_df.reset_index(inplace=True)

    return


def mil_davis_activity(data, params):
    inv_df = data.groupby('date').apply(find_temperature_inversions, params)
    inv_df['theta_top']
    inv_df['theta_bottom'] 
    # Need pressure at inversion top to be able to convert to potential temperature.
    # Check paper - I think it's average weighted by inversion thickness
    # With potential temperature precalculated i.e. in the IGRA2 derived this can be adjusted easily

#def lower_tropospheric_stability(data, params):
    
