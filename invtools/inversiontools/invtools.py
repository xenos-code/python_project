"""Method for identifying temperature inversion layers in a radiosonde sounding.
The core of the method is to fit a piecewise linear function to the profile, then 
estimate the 1-alpha confidence intervals around the slope for each piece. Pieces 
that have a positive slope at the alpha significance level are merged. The method 
is designed to work with GRUAN netcdf radiosonde data, which includes uncertainty
information for each measurement.

GRUAN data processing includes a low-pass filter that reduces the temporal
resolution of temperature from 1s to 10s, but leaves the data in 1s resolution 
format. As the ascent speed is approximately 5 m/s, this means that the effective 
vertical resolution for temperature is 50m. For relative humidity, wind, and altitude,
the vertical resolution is different. (Dirksen et al., 2014). The effective temporal
resolution of the relative humidity depends on temperature (and other things) and as
such is included as its own column. Below the tropopause, the resolution is 10s. Altitude
resolution is 15 seconds, and wind is much lower resolution, at approximately 40s.
Prior to fitting the linear piecewise function, we interpolate to a regular 50 m grid 
starting at ground level. The grid is represented by the parameter dz, and so that 
sensitivity checks can be run is left as a user-specifiable parameter.

Uncertainty in dewpoint temperature: I'm still working on recreating the results of
the paper, but in the meantime, I use the min/max method to get an upper estimate on
the uncertainty. For Td(Ta, RH), the estimate is 
$$U_{T_d} = 1/2*(T_d(T_a+U_{T_a}, RH+U_{RH}) - T_d(T_a-U_{T_a}, RH-U_{RH})$$

compute_inversions:
    applies inversion algorithms to specified netcdf files, including classification.
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
import metpy.calc as mcalc
from metpy.units import units
import warnings

warnings.filterwarnings('ignore', 'invalid value encountered in less', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in greater', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in multiply', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in true_divide', RuntimeWarning)
warnings.filterwarnings('ignore', 'divide by zero encountered in true_divide', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in log', RuntimeWarning)

def default_params():
    """Default set of parameters. Parameters let the other functions
    know what column names are as well as set options for the classification
    of temperature inversions.
    TODO: Add units, at least in description of default_params
    TODO: make it so that it doesn't fail if uncertainty data is missing.
    """
    
    return {'temperature': 'temperature',
           'height': 'altitude',
           'pressure': 'pressure',
           'potential_temperature': None,
           'time': 'date',
           'u_temperature': 'uncertainty_temperature',
           'u_height': 'uncertainty_altitude',
           'dewpoint_temperature': 'dewpoint_temperature',
           'u_dewpoint_temperature': 'uncertainty_dewpoint_temperature',
           'iso_base': False,
           'iso_above': True,
           'iso_within': True,
           'iso_alone': False,
           'first_only': False,
           'na_tol': 0.1,
           'max_embed_depth': 0,
           'min_inv_depth': 10,
           'min_lapse_rate': 0,
           'max_lapse_rate': 0.4,
           'classify_inversions': True,
           'sbi_threshold': 0, # heighest inversion base for sbi can be. only used for classification
           'tmax_inversion': True, # If true, append tmax inversion to inversion dataframe
           'lts': True, # If true, append LTS to inversion dataframe
           'k': 2, # Multiples of sigma for uncertainty. 2sigma = 95% confidence.
           'dz': 10 # Could set it up so that dz is also potentially a vector
           }

def default_igra_params():
    """Convenience function to set parameters for IGRA2 profiles in inversion metrics paper."""
    params = default_params()
    params['height'] = 'calculated_height'
    params['dewpoint_temperature'] = None
    params['classify_inversions'] = False
    params['min_inv_depth'] = 50
    params['max_embed_depth'] = 50
    params['iso_above'] = True
    params['k'] = 0
    params['lts'] = False
    params['lts_top'] = 925
    params['tmax_inversion'] = False
    params['min_lapse_rate'] = 0
    return params

def default_cloud_params(params):
    """Convenience function to set parameters for IGRA2 cloud layer identification"""
    params['minRH'] = 85 # percent. RH threshold at top of cloud
    params['maxRH'] = 90 # percent. RH threshold at base of cloud
    params['intRH'] = 80 # percent. Allowed low RH between merged cloud layers
    params['intZ'] = 300 # meters. Allowed distance between merged cloud layers
    
    return params
    
def build_layer_df(sign_vector, data, params):
    """Selects the data at the top and bottom of layers of constant sign
    based on the sign vector, differences the data across the layers, and
    adds columns 'layer_strength' and 'layer_depth'."""
    idx_list = np.atleast_1d(np.argwhere(
        sign_vector[1:] != sign_vector[:-1]).squeeze())

    if sign_vector[0] == 1:
        idx_list = np.hstack([[0], idx_list])

    idxb = np.array(idx_list[:-1])
    idxt = np.array(idx_list[1:])
    layer_dict = {'idx_base': idxb,
                  'idx_top': idxt}

    for cc in [cc for cc in data.columns if (cc != 'date')]:
        layer_dict[cc + '_base'] = data.loc[idxb, cc].values
        layer_dict[cc + '_top'] = data.loc[idxt, cc].values

    layer_dict['layer_strength'] = layer_dict[params['temperature'] + '_top'] - layer_dict[params['temperature'] + '_base']
    layer_dict['layer_depth'] = layer_dict[params['height'] + '_top'] - layer_dict[params['height'] + '_base']

    layer_df = pd.DataFrame(layer_dict)
    layer_df = layer_df.dropna(subset=['layer_strength']).reset_index(drop=True)
    if len(layer_df) == 0:
        layer_df = pd.DataFrame(columns = layer_df.columns, index=[0])
    layer_df['date'] = np.array(data.date.values[0])    
    
    return layer_df

def merge_iso_layers(sign_vector, params):
    """Based on settings in params, add isothermal
    layers to inversion layers."""

    # Apply rules for including isothermal layers
    if params['iso_alone']:
        sign_vector[sign_vector == 0] = 1
        return sign_vector

    if params['iso_base']:
        iso_idx = (sign_vector[:-1] == 0) & (sign_vector[1:] == 1)
        while np.any(iso_idx):
            sign_vector[:-1][iso_idx] = 1
            iso_idx = (sign_vector[:-1] == 0) & (sign_vector[1:] == 1)

    if params['iso_above']:
        iso_idx = (sign_vector[:-1] == 1) & (sign_vector[1:] == 0)
        while np.any(iso_idx):
            sign_vector[1:][iso_idx] = 1
            iso_idx = (sign_vector[:-1] == 1) & (sign_vector[1:] == 0)

    if not params['iso_alone']:
        sign_vector[sign_vector==0] = -1

    return sign_vector

def drop_thin_inv_layers(sign_vector, layer_df, params):
    """Inversion layers thinner than the minimum inversion
    depth are dropped. If minimum inversion depth is smaller
    than dz, then this code doesn't do anything.
    
    Alternatively, layers could be dropped based on statistical significance 
    rather than a depth threshold.
    """

    thin_layers = ((layer_df['layer_strength'] > 0) &
                   (layer_df['layer_depth'] < params['min_inv_depth'])).values
    
    if np.any(thin_layers):
        for x in np.argwhere(thin_layers):
            idxb = int(layer_df.loc[x, 'idx_base']) + 1
            idxt = int(layer_df.loc[x, 'idx_top'])
            sign_vector[idxb:idxt] = -1
    return sign_vector

def merge_neg_layers(sign_vector, layer_df, params):
    """Thin layers with negative lapse rate embedded within
    inversion layers are counted as part of the inversion layer."""

    thin_layers = ((layer_df['layer_strength'] < 0) &
                   (layer_df['layer_depth'] < params['max_embed_depth'])).values
    if np.any(thin_layers):
        for x in np.argwhere(thin_layers):
            if (x > 0) & (x < len(layer_df)-1):
                if np.all([layer_df.loc[x-1, 'layer_strength'].values > 0,
                           layer_df.loc[x+1, 'layer_strength'].values > 0]):
                    idxb = int(layer_df.loc[x, 'idx_base']) + 1
                    idxt = int(layer_df.loc[x, 'idx_top'])
                    sign_vector[idxb:idxt] = 1
    return sign_vector

    
def build_inversion_df(layer_df, params):
    """Take final layer df and add uncertainty values and classification"""
    
    layer_df['layer_lapse_rate'] = layer_df['layer_strength']/layer_df['layer_depth']
    if params['dewpoint_temperature'] is not None:
        layer_df['dewpoint_change'] = (layer_df['dewpoint_temperature_top'] - 
                                        layer_df['dewpoint_temperature_base'])
       

    if params['k'] > 0:
        # get uncertainty of inversion strength and depth
        # if available.
        u_tt = layer_df['uncertainty_temperature_top']
        u_tb = layer_df['uncertainty_temperature_base']

        u_zt = layer_df['uncertainty_altitude_top']
        u_zb = layer_df['uncertainty_altitude_base']

        u_tdt = layer_df['uncertainty_dewpoint_temperature_top']
        u_tdb = layer_df['uncertainty_dewpoint_temperature_base']
    
        layer_df['uncertainty_layer_strength'] = (u_tt**2 + u_tb**2)**0.5
        layer_df['uncertainty_layer_depth'] = (u_zt**2 + u_zb**2)**0.5
        layer_df['uncertainty_layer_dewpoint_change'] = (u_tdt**2 + u_tdb**2)**0.5

        layer_df['uncertainty_layer_lapse_rate'] = (layer_df['layer_lapse_rate']**2 * ((
            layer_df['uncertainty_layer_strength']/layer_df['layer_strength'])**2 +
            (layer_df['uncertainty_layer_depth']/layer_df['layer_depth'])**2))**0.5

    # Inversion classification method subject to change
    if params['classify_inversions']:
        layer_df['classification'] = np.nan
        d_td = layer_df['dewpoint_change']
        u_d_td = layer_df['uncertainty_layer_dewpoint_change']
        
        k = params['k']
              
        layer_df.loc[(d_td - k * u_d_td) > 0,
                     'classification'] = 'EI-WAA'
        
        layer_df.loc[(d_td + k * u_d_td) < 0,
                     'classification'] = 'EI-AC'

        layer_df.loc[((d_td - k * u_d_td) < 0) & ((d_td + k * u_d_td) > 0),
                     'classification'] = 'EI-U'

        layer_df.loc[layer_df['altitude_base'] <= params['sbi_threshold'], 'classification'] = 'SBI'
        
    if params['k'] > 0:
        significant_inversion = layer_df['layer_lapse_rate'] - params['k']*layer_df['uncertainty_layer_lapse_rate'] > params['min_lapse_rate']
        layer_df = layer_df.loc[significant_inversion].copy()
    else:
        layer_df = layer_df.loc[layer_df['layer_lapse_rate'] > params['min_lapse_rate'] ,:].copy()
    
    if len(layer_df) > 0:
        layer_df.reset_index(drop=True, inplace=True)
        layer_df.index += 1
        return layer_df
    else:
        # hopefully this makes it have at least a single line
        layer_df = pd.DataFrame(columns=layer_df.columns, index=[0])
     
        return layer_df
    
def find_inversions(data, params):
    """Given a dataframe with a single sounding, find all inversion layers."""
    data = data.copy()
    if 'level_0' not in data:
        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)
        
    if params['tmax_inversion']:
        """Identify the temperature maximum as the inversion top,
        and return that instead."""
        # TO DO: set up parameter to turn the uncertainty-aware feature on or off
        idx_tmax = data[params['temperature']].idxmax()
        idx_top = idx_tmax
        
        idx_base = 0
        sign_vector = np.ones(len(data)) * np.nan
        sign_vector[idx_base:idx_top+1] = 1
        sign_vector[idx_top+1:] = -1

        tm_layer_df_a = build_layer_df(sign_vector, data, params)
        tm_layer_df_b = build_inversion_df(tm_layer_df_a, params)
        
        idx = int(tm_layer_df_a.layer_strength > 0)

        tm_layer_df = pd.DataFrame(columns=tm_layer_df_b.columns,
                                   index=[idx])
        
        tm_layer_df['algorithm'] = 'tm'
        
        for col in tm_layer_df_a.columns:
            tm_layer_df.loc[idx, col] = tm_layer_df_a.loc[0, col]
        
    else:
        tm_layer_df = None
        
    if params['lts']:
        """Problem here is that LTS is evaluated at exact pressure levels, which may not exist in the sounding.
        Those can be interpolated from the interpolated data, but it messes up the build_inversion_df function
        since those need a sign vector."""
        
        idx_base = 0
        # could set it to extrapolate to 1000 mb here.
        # could also set the lts top height here
        p0 = np.max(data.pressure)
        data_new = xr.Dataset(data.set_index('pressure')).interp(
            {'pressure': [p0, 925, 850]}).to_dataframe()
        data_new.reset_index(inplace=True)
        data_new['date'] = data['date'].values[0]
        sign_vector = np.array([-1, 1, -1, -1])
        if params['lts_top'] == 850:
            sign_vector = np.array([-1, 1, 1, -1])            
        lts_layer_df_a = build_layer_df(sign_vector, data_new, params)
        lts_layer_df_b = build_inversion_df(lts_layer_df_a, params)
        
        idx = int(lts_layer_df_a.layer_strength > 0)
        lts_layer_df = pd.DataFrame(columns=lts_layer_df_b.columns,
                                   index=[idx])
        lts_layer_df.loc[idx, 'algorithm'] = 'lts'
        for col in lts_layer_df_a.columns:
            lts_layer_df.loc[idx, col] = lts_layer_df_a.loc[0, col]
    
    else:
        lts_layer_df = None

    # Forward difference, in line with previous studies
    dt = np.diff(data[params['temperature']])
    dz = np.diff(data[params['height']])
    #ut = data[params['u_temperature']].values[:-1]
    #uz = data[params['u_height']].values[:-1]

    lr = dt/dz
    #ulr = np.hstack([np.sqrt(lr**2 * ((ut/dt)**2 + (uz/dz)**2)), np.nan])
    lr = np.hstack([lr, np.nan])
    k = params['k']
    
    
    # Since the initial data is over-resolved, the uncertainty
    # is very high. So for the first pass, we don't use the estimated
    # uncertainty.
    sign_vector = np.ones(len(data)) * np.nan

    sign_vector[lr > 0] = 1
    sign_vector[lr < 0] = -1
    #sign_vector[((lr - k*ulr) < 0) & ((lr + k*ulr) > 0)] = 0
    sign_vector = np.hstack(([-1], sign_vector[:-1]))    

    updating = True
    layer_df = build_layer_df(sign_vector, data, params)
    if len(layer_df) == 1:
        updating = False
        
    while updating:
        # Move the lapse rate check thing into the layer df build part?
        sign_vector = merge_iso_layers(sign_vector, params)
        sign_vector = drop_thin_inv_layers(sign_vector, layer_df, params)
        sign_vector = merge_neg_layers(sign_vector, layer_df, params)
        
        new_layer_df = build_layer_df(sign_vector, data, params)

        if layer_df.shape == new_layer_df.shape:
            updating = False
        layer_df = new_layer_df
    
    
    layer_df = build_inversion_df(layer_df, params)
    
    if params['first_only']:
        if len(layer_df) > 1:
            data_val = layer_df.loc[1,:].values
            layer_df = pd.DataFrame(columns=layer_df.columns,
                                        index=[1])
            layer_df.loc[1,:] = data_val
            layer_df['algorithm'] = 'k'
        else:
            layer_df['algorithm'] = 'all'
    else:
        layer_df['algorithm'] = 'all'
        
    layer_df =  pd.concat([layer_df, tm_layer_df, lts_layer_df])
    layer_df.drop(['idx_base','idx_top'], axis=1, inplace=True)
    if 'latitude_base' in layer_df:
        layer_df.rename({'latitude_base': 'latitude',
                         'longitude_base': 'longitude'}, axis=1, inplace=True)
        layer_df.drop(['latitude_top', 'longitude_top'], axis=1, inplace=True)
    return layer_df


def find_clouds(data, params):
    """Identify cloud layers using the RH method"""
    
    
    
    return