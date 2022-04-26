"""Tool for identifying inversions in a dataframe containing one atmospheric sounding."""
# To do: implement units check for thresholds. I'm not sure how to access the units of a dataarray.
# To do: what happens if index_list is empty?
from .core import build_layer_df, merge_layers
from metpy.units import units
import metpy.calc as mcalc
import numpy as np
import pandas as pd

def invfinder(data, params={'max_embed_depth': 100, 
                            'min_dz': 0, #* units('m'),
                            'min_dp': 20,# * units('hPa'),
                            'min_dt': 2.5,# * units('K'),
                            'min_drh': 20,# * units('percent'),
                            'rh_or_dt': True}):
    """Implementation of inversion finder that returns multiple inversion layers.
    At the moment the data on units doesn't fully come through, so that needs to get fixed.
    """
    def init_index_list(data):
        """Uses the first difference to find regions of 
        nonnegative slope. """

        dt = data.temperature.shift({'index':-1}) - data.temperature
        dt = list(dt.values >= 0)
        index_list, = np.nonzero(np.diff(dt))
        
        index_list = [x + 1 for x in index_list]
        if len(index_list) > 0:
            if index_list[0] == 0:
                index_list = list(index_list) + [len(dt)-1]
            else:
                index_list = [0] + list(index_list) + [len(dt)-1]
        else:
            index_list = [0, len(dt)-1]
            
        return index_list


    index_list = init_index_list(data)

    
    while True:
        init_length = len(index_list)
        layer_df = build_layer_df(index_list, data)
        negative_lapse = layer_df['temperature_top'] - layer_df['temperature_base'] < 0
        # could do layer_df.loc[negative_lapse,:] instead
        merge_layers(index_list, layer_df.loc[negative_lapse,:], 'height', params['max_embed_depth'], upper=True)
        if len(index_list) == init_length:
            break
            
    layer_df = layer_df.loc[
        layer_df['temperature_top'] - layer_df['temperature_base'] > 0,:] #.reset_index(drop=True)
    zdepth = (layer_df['height_top'] - layer_df['height_base'])# * data.height.units
    pdepth = (layer_df['pressure_base'] - layer_df['pressure_top']) #* data.pressure.units
    tstren = (layer_df['temperature_top'] - layer_df['temperature_base'])# * data.temperature.units # make delta temperature
    hstren = np.abs(layer_df['relative_humidity_top'] - layer_df['relative_humidity_base'])# * data.relative_humidity.units
    
    zdepth_check = zdepth > params['min_dz']
    pdepth_check = pdepth > params['min_dp']
    tstren_check = tstren > params['min_dt']
    hstren_check = hstren > params['min_drh']
    if params['rh_or_dt']:
        idx_sel = (zdepth_check & pdepth_check) & (tstren_check | hstren_check)
    else:
        idx_sel = (zdepth_check & pdepth_check) & (tstren_check & hstren_check)
    
    layer_df = layer_df.loc[idx_sel, :].reset_index(drop=True)
    layer_df.index = pd.Index(layer_df.index.values + 1, name='inv_number') # change to arange
    if len(layer_df) == 0:
        layer_df.loc[0,:] = np.nan
        layer_df.loc[0,'date'] = data.sel(index=0)['date'].values
    return layer_df