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
warnings.filterwarnings('ignore', 'invalid value encountered in true_divide', RuntimeWarning)
warnings.filterwarnings('ignore', 'divide by zero encountered in true_divide', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in log', RuntimeWarning)

def default_params():
    """Default set of parameters. Parameters let the other functions
    know what column names are as well as set options for the classification
    of temperature inversions.
    TODO: Add units, at least in description of default_params
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
           'iso_above': False,
           'iso_within': True,
           'iso_alone': False,
           'na_tol': 0.1,
           'max_embed_depth': 0,
           'min_inv_depth': 10,
           'max_lapse_rate': 0.4,
           'classify_inversions': True,
           'sbi_threshold': 0, 
           'tmax_inversion': False, # If true, append tmax inversion to inversion dataframe
           'lts': False, # If true, append LTS to inversion dataframe
           'k': 2,
           'dz': 10 # Could set it up so that dz is also potentially a vector
           }

def process_gruan_sounding(filepath, params):
    """Rename variables and compute dewpoint temperatures for the
    dataset located at filepath. Adds lapse rate also.
    
    To do: adjust to work with variable dz.
    """
    
    varnames = {'time': 'time',
                 'pressure': 'press',
                 'temperature': 'temp',
                 'relative_humidity': 'rh',
                 'wind_direction': 'wdir',
                 'wind_speed': 'wspeed',
                 'geopotential_height': 'geopot',
                 'longitude': 'lon',
                 'latitude': 'lat',
                 'altitude': 'alt',
                 'u_wind': 'u',
                 'v_wind': 'v',
                 'frost_point': 'FP',
                 'water_vapor_mixing_ratio': 'WVMR',
                 'ascent_descent_speed': 'asc',
                 'shortwave_radiation': 'SWrad',
                 'uncertainty_shortwave_radiation': 'u_SWrad',
                 'air_temperature_correlation': 'cor_temp',
                 'uncertainty_air_temperature_correlation': 'u_cor_temp',
                 'standard_deviation_of_air_temperature': 'u_std_temp',
                 'uncertainty_temperature': 'u_temp',
                 'uncertainty_altitude': 'u_alt',
                 'uncertainty_pressure': 'u_press',
                 'resolution_relative_humidity': 'res_rh',
                 'standard_deviation_of_relative_humidity': 'u_std_rh',
                 'relative_humidity_correlation': 'cor_rh',
                 'uncertainty_relative_humidity_correlation': 'u_cor_rh',
                 'uncertainty_relative_humidity': 'u_rh',
                 'uncertainty_wind_direction': 'u_wdir',
                 'uncertainty_wind_speed': 'u_wspeed'}
    
    variables = ['pressure', 'temperature', 'relative_humidity', 'u_wind', 'v_wind',
             'uncertainty_pressure', 'uncertainty_temperature', 'uncertainty_relative_humidity',
             'uncertainty_altitude', 'uncertainty_wind_speed', 'uncertainty_wind_direction']

    with xr.open_dataset(filepath) as ds:
        ds = ds.dropna('time')
        dz = params['dz']
        minz = ds.variables[varnames['geopotential_height']].min()
        altitude = (ds.variables[varnames['geopotential_height']] - minz).values

        data_vars = {variable: ('altitude', 
                                ds.variables[varnames[variable]]) for variable in variables}
        rh = data_vars['relative_humidity'][1].values * units.dimensionless
        t = data_vars['temperature'][1].values * units.kelvin
        urh = data_vars['uncertainty_relative_humidity'][1].values * units.dimensionless
        ut = data_vars['uncertainty_temperature'][1].values * units.kelvin
        td = mcalc.dewpoint_from_relative_humidity(t, rh).to_base_units().to_tuple()[0]
        upper_td = mcalc.dewpoint_from_relative_humidity(t + ut, 
                                                         rh + urh).to_base_units().to_tuple()[0]
        lower_td = mcalc.dewpoint_from_relative_humidity(t - ut,
                                                         rh - urh).to_base_units().to_tuple()[0]
        utd = 1./2*(upper_td - lower_td)

        data_vars['dewpoint_temperature'] = ('altitude', xr.DataArray(data = td,
                                                         dims = {'altitude': altitude},
                                                         name = 'dewpoint_temperature'))
        data_vars['uncertainty_dewpoint_temperature'] = ('altitude', 
                                                         xr.DataArray(data = utd,
                                                            dims = {'altitude': altitude},
                                                  name = 'uncertainty_dewpoint_temperature'))
        ordered_keys = ['pressure',
                         'temperature',
                         'dewpoint_temperature',
                         'relative_humidity',
                         'u_wind',
                         'v_wind',
                         'uncertainty_altitude',
                         'uncertainty_dewpoint_temperature',
                         'uncertainty_pressure',
                         'uncertainty_relative_humidity',
                         'uncertainty_temperature',
                         'uncertainty_wind_direction',
                         'uncertainty_wind_speed']
        
        data_vars = {variable: data_vars[variable] for variable in ordered_keys}
        ds1 = xr.Dataset(data_vars,
                        coords={'altitude': altitude},
                        attrs=ds.attrs)

        # Make sure that attributes get saved
        for variable in ordered_keys:
            ds1.variables[variable].attrs = data_vars[variable][1].attrs

        # Add in attributes for computed variables
        ds1.variables['dewpoint_temperature'].attrs = {'standard_name': 'dewpoint_temperature',
                                                       'units': 'kelvin',
                                                       'long_name': 'dewpoint_temperature',
                                                       'comment': 'Calculated via metpy.calc.dewpoint_from_relative_humidity',
                                                       'related_columns': 'uncertainty_dewpoint_temperature'}
        ds1.variables['uncertainty_dewpoint_temperature'].attrs = {'standard_name': 'dewpoint_temperature standard_error',
                                                       'units': 'kelvin',
                                                       'long_name': 'Uncertainty of dewpoint_temperature',
                                                       'comment': 'Min-max uncertainty (k=1) of total uncertainty in dewpoint temperature.',
                                                       'related_columns': 'uncertainty_dewpoint_temperature'}
        ds1.coords['altitude'].attrs = {
            'standard_name': 'altitude',
            'units': 'm',
            'long_name': 'geopotential height above ground level',
            'assumed_ground_level': str(np.round(minz, 2).values) + ' m'}
        
        ds_interp = add_lapse_rate2(ds1.interp(coords={'altitude': np.arange(0,5000, params['dz'])}), params)
        del ds1
        
    dt = ds_interp.diff(dim=params['height'], n=1, label='lower').variables[params['temperature']]

    u_dt = ((ds_interp.variables[params['u_temperature']].values[1:]**2 +
                  ds_interp.variables[params['u_temperature']].values[0:-1]**2)**0.5)

    u_dz = ((ds_interp.variables[params['u_height']].values[1:]**2 +
                  ds_interp.variables[params['u_height']].values[0:-1]**2)**0.5)

    n = ds_interp.coords[params['height']].values.size

    dt_dz = np.full((n,), np.nan)
    u_dt_dz = np.full((n,), np.nan)
    dt_dz[:-1] = dt.values/dz
    u_dt_dz[:-1] = ((dt/dz)**2 * ((u_dt/dt)**2 + (u_dz/dz)**2))**0.5
    dt_dz = xr.DataArray(dt_dz, dims={params['height']: ds_interp.coords[params['height']]})
    u_dt_dz = xr.DataArray(u_dt_dz, dims={params['height']: ds_interp.coords[params['height']]}) 

    
    dtd = ds_interp.diff(dim=params['height'], n=1, label='lower').variables[params['dewpoint_temperature']]
    u_dtd = ((ds_interp.variables[params['u_dewpoint_temperature']].values[1:]**2 +
              ds_interp.variables[params['u_dewpoint_temperature']].values[0:-1]**2)**0.5)
    dtd_dz = np.full((n,), np.nan)
    u_dtd_dz = np.full((n,), np.nan)
    dtd_dz[:-1] = dtd.values/dz
    u_dtd_dz[:-1] = ((dt/dz)**2 * ((u_dt/dt)**2 + (u_dz/dz)**2))**0.5
    dtd_dz = xr.DataArray(dtd_dz, dims={params['height']: ds_interp.coords[params['height']]})
    u_dtd_dz = xr.DataArray(u_dtd_dz, dims={params['height']: ds_interp.coords[params['height']]}) 
    return ds_interp.assign({'lapse_rate': dt_dz,
                           'dewpoint_lapse_rate': dtd_dz,
                           'uncertainty_lapse_rate': u_dt_dz,
                           'uncertainty_dewpoint_lapse_rate': u_dtd_dz})

    
def add_lapse_rate(dataset, params, dewpoint=False):
    """Computes the lapse rate and returns a dataset with lapse rate added
    along with a column with the k=1 uncertainty for the lapse rate. Assumes
    that the altitude or height is provided as the vertical dimension of the
    dataset. Note that here the uncertainty is the uncertainty of the slope of
    the piecewise linear function defined by the sounding data, not the uncertainty
    in the forward difference estimate of the derivative."""
    
    # Note: may need to switch to backward difference to avoid 
    # inconsistency with inversion definitions?
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    # step one: apply forward difference
    dt = dataset.diff(dim=params['height'], n=1, label='lower').variables[params['temperature']]
    dz = params['dz'] # Here, can alter to use variable spacing if needed. Might be better if
                      # this code is to be used for inversion strength also
    
    u_dt = ((dataset.variables[params['u_temperature']].shift(shifts={params['height']: 1})**2 + 
            dataset.variables[params['u_temperature']]**2)**0.5)[:,0:-1].values

    u_dz = ((dataset.variables[params['u_height']].shift(shifts={params['height']: 1})**2 + 
            dataset.variables[params['u_height']]**2)**0.5)[:,0:-1].values

    n = dataset.coords[params['height']].values.size
    m = dataset.coords[params['time']].values.size

    dt_dz = np.full((m, n), np.nan)
    u_dt_dz = np.full((m, n), np.nan)
    dt_dz[:, :-1] = dt.values/dz
    u_dt_dz[:, :-1] = ((dt/dz)**2 * ((u_dt/dt)**2 + (u_dz/dz)**2))**0.5
    dt_dz = xr.DataArray(dt_dz, dims={params['time']: dataset.coords[params['time']],
                                        params['height']: dataset.coords[params['height']]})
    u_dt_dz = xr.DataArray(u_dt_dz, dims={params['time']: dataset.coords[params['time']],
                                        params['height']: dataset.coords[params['height']]}) 
    if dewpoint:
        dtd = dataset.diff(dim=params['height'], n=1, label='lower').variables[params['dewpoint_temperature']]
        u_dtd = ((dataset.variables[params['u_dewpoint_temperature']].shift(shifts={params['height']: 1})**2 + 
                dataset.variables[params['u_dewpoint_temperature']]**2)**0.5)[:,0:-1].values
        dtd_dz = np.full((m, n), np.nan)
        u_dtd_dz = np.full((m, n), np.nan)
        dtd_dz[:, :-1] = dtd.values/dz
        u_dtd_dz[:, :-1] = ((dt/dz)**2 * ((u_dt/dt)**2 + (u_dz/dz)**2))**0.5
        dtd_dz = xr.DataArray(dtd_dz, dims={params['time']: dataset.coords[params['time']],
                                            params['height']: dataset.coords[params['height']]})
        u_dtd_dz = xr.DataArray(u_dtd_dz, dims={params['time']: dataset.coords[params['time']],
                                            params['height']: dataset.coords[params['height']]}) 
        return dataset.assign({'lapse_rate': dt_dz,
                               'dewpoint_lapse_rate': dtd_dz,
                               'uncertainty_lapse_rate': u_dt_dz,
                               'uncertainty_dewpoint_lapse_rate': u_dtd_dz})
    else:
        return dataset.assign({'lapse_rate': dt_dz,
                           'uncertainty_lapse_rate': u_dt_dz})

def add_lapse_rate2(dataset, params, dewpoint=False):
    """Computes the lapse rate and returns a dataset with lapse rate added
    along with a column with the k=1 uncertainty for the lapse rate. Assumes
    that the altitude or height is provided as the vertical dimension of the
    dataset. Note that here the uncertainty is the uncertainty of the slope of
    the piecewise linear function defined by the sounding data, not the uncertainty
    in the forward difference estimate of the derivative.
    
    This is a downgrade of add_lapse_rate in that it only does one-dimensional datasets,
    since I'm working out an issue with having duplicate dates.
    """
    
    # Note: may need to switch to backward difference to avoid 
    # inconsistency with inversion definitions?
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    # step one: apply forward difference
    dt = dataset.diff(dim=params['height'], n=1, label='lower').variables[params['temperature']]
    dz = params['dz'] # Here, can alter to use variable spacing if needed. Might be better if
                      # this code is to be used for inversion strength also
    
    u_dt = ((dataset.variables[params['u_temperature']].values[1:]**2 +
                  dataset.variables[params['u_temperature']].values[0:-1]**2)**0.5)

    u_dz = ((dataset.variables[params['u_height']].values[1:]**2 +
                  dataset.variables[params['u_height']].values[0:-1]**2)**0.5)
    
    n = dataset.coords[params['height']].values.size

    dt_dz = np.full((n,), np.nan)
    u_dt_dz = np.full((n,), np.nan)
    dt_dz[:-1] = dt.values/dz
    u_dt_dz[:-1] = ((dt/dz)**2 * ((u_dt/dt)**2 + (u_dz/dz)**2))**0.5
    dt_dz = xr.DataArray(dt_dz, dims={params['height']: dataset.coords[params['height']]})
    u_dt_dz = xr.DataArray(u_dt_dz, dims={params['height']: dataset.coords[params['height']]}) 
    
    if dewpoint:
        dtd = dataset.diff(dim=params['height'], n=1, label='lower').variables[params['dewpoint_temperature']]
        u_dtd = ((dataset.variables[params['u_dewpoint_temperature']].values[1:]**2 +
                  dataset.variables[params['u_dewpoint_temperature']].values[0:-1]**2)**0.5)
        dtd_dz = np.full((n,), np.nan)
        u_dtd_dz = np.full((n,), np.nan)
        dtd_dz[:-1] = dtd.values/dz
        u_dtd_dz[:-1] = ((dt/dz)**2 * ((u_dt/dt)**2 + (u_dz/dz)**2))**0.5
        dtd_dz = xr.DataArray(dtd_dz, dims={params['height']: dataset.coords[params['height']]})
        u_dtd_dz = xr.DataArray(u_dtd_dz, dims={params['height']: dataset.coords[params['height']]}) 
        return dataset.assign({'lapse_rate': dt_dz,
                               'dewpoint_lapse_rate': dtd_dz,
                               'uncertainty_lapse_rate': u_dt_dz,
                               'uncertainty_dewpoint_lapse_rate': u_dtd_dz})
    else:
        return dataset.assign({'lapse_rate': dt_dz,
                           'uncertainty_lapse_rate': u_dt_dz})




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
#       # I'll have to figure this part out in a bit,  
#         if len(idx_list) % 2 == 1:
#             idx_list = np.concatenate((idx_list, [len(sign_vector)-1]))

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
    than dz, then this code doesn't do anything."""

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
    
    layer_df['dewpoint_change'] = (layer_df['dewpoint_temperature_top'] - 
                                    layer_df['dewpoint_temperature_base'])

    # get uncertainty of inversion strength and depth
    u_tt = layer_df['uncertainty_temperature_top']
    u_tb = layer_df['uncertainty_temperature_base']

    u_zt = layer_df['uncertainty_altitude_top']
    u_zb = layer_df['uncertainty_altitude_base']

    u_tdt = layer_df['uncertainty_dewpoint_temperature_top']
    u_tdb = layer_df['uncertainty_dewpoint_temperature_base']
    
    layer_df['uncertainty_layer_strength'] = (u_tt**2 + u_tb**2)**0.5
    layer_df['uncertainty_layer_depth'] = (u_zt**2 + u_zb**2)**0.5
    layer_df['uncertainty_layer_dewpoint_change'] = (u_tdt**2 + u_tdb**2)**0.5


    layer_df['layer_lapse_rate'] = layer_df['layer_strength']/layer_df['layer_depth']
    layer_df['uncertainty_layer_lapse_rate'] = (layer_df['layer_lapse_rate']**2 * ((
                                                layer_df['uncertainty_layer_strength']/layer_df['layer_strength'])**2 +
                                                (layer_df['uncertainty_layer_depth']/layer_df['layer_depth'])**2))**0.5

    if params['classify_inversions']:
        layer_df['classification'] = np.nan
        d_td = layer_df['dewpoint_change']
        u_d_td = layer_df['uncertainty_layer_dewpoint_change']
        
        k = params['k']
              
        layer_df.loc[(d_td - k * u_d_td > 0),
                     'classification'] = 'EI-WAA'
        
        layer_df.loc[(d_td + k * u_d_td < 0),
                     'classification'] = 'EI-WAA'

        layer_df.loc[(d_td - k * u_d_td < 0) & (d_td + k * u_d_td > 0),
                     'classification'] = 'EI-U'

        layer_df.loc[layer_df['altitude_base'] <= params['sbi_threshold'], 'classification'] = 'SBI'
    significant_inversion = layer_df['layer_lapse_rate'] - params['k']*layer_df['uncertainty_layer_lapse_rate'] > 0

    layer_df.drop(['lapse_rate_top', 'lapse_rate_base',
                   'uncertainty_lapse_rate_top', 
                   'uncertainty_lapse_rate_base'], axis=1, inplace=True)
    layer_df = layer_df.loc[significant_inversion]
    if len(layer_df) > 0:
        layer_df.reset_index(drop=True, inplace=True)
        layer_df.index += 1
        return layer_df
    else:
        # hopefully this makes it have at least a single line
        layer_df['altitude_base'] = np.nan
        return layer_df
    
def find_inversions(data, params):
    """Given a dataframe with a single sounding, find all inversion layers."""

    if 'level: 0' not in data:
        data.reset_index(inplace=True)

    if params['tmax_inversion']:
        """Identify the temperature maximum as the inversion top,
        and return that instead."""
        idx_top = data[params['temperature']].idxmax()
        idx_base = 0
        sign_vector = np.ones(len(data)) * np.nan
        sign_vector[idx_base:idx_top+1] = 1
        sign_vector[idx_top+1:] = -1
        tm_layer_df = build_layer_df(sign_vector, data, params)
        tm_layer_df = build_inversion_df(tm_layer_df, params)
        tm_layer_df.classification = 'tm'
        return tm_layer_df    
        
    if params['lts']:
        """Problem here is that LTS is evaluated at exact pressure levels, which may not exist in the sounding.
        Those can be interpolated from the interpolated data, but it messes up the build_inversion_df function
        since those need a sign vector."""
        
        idx_base = 0
        p0 = np.max(data.pressure)
        data_new = xr.Dataset(data.set_index('pressure')).interp(
            {'pressure': [p0, 925]}).to_dataframe()
        data_new.reset_index(inplace=True)
        data_new['date'] = data['date'].values[0]
        sign_vector = np.array([-1, 1, -1])
        lts_layer_df = build_layer_df(sign_vector, data_new, params)
        lts_layer_df = build_inversion_df(lts_layer_df, params)
        lts_layer_df['classification'] = 'lts'
        return lts_layer_df
          
        
    lr = data['lapse_rate'].values
    ulr = data['uncertainty_lapse_rate'].values
    k = params['k']
    
    sign_vector = np.ones(len(data)) * np.nan

    sign_vector[(lr - k*ulr) > 0] = 1
    sign_vector[(lr + k*ulr) < 0] = -1
    sign_vector[((lr - k*ulr) < 0) & ((lr + k*ulr) > 0)] = 0
    sign_vector = np.hstack(([-1], sign_vector[:-1]))    
    
    updating = True
    layer_df = build_layer_df(sign_vector, data, params)
    if len(layer_df) == 1:
        updating = False
        
    while updating:
        sign_vector = merge_iso_layers(sign_vector, params)
        sign_vector = drop_thin_inv_layers(sign_vector, layer_df, params)
        sign_vector = merge_neg_layers(sign_vector, layer_df, params)
        
        new_layer_df = build_layer_df(sign_vector, data, params)

        if layer_df.shape == new_layer_df.shape:
            updating = False
        layer_df = new_layer_df
    
    
    layer_df = build_inversion_df(layer_df, params)



      
        
    return layer_df


