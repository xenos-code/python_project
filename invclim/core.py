"""Utilities used by the main functions in the module."""
import numpy
import pandas

def setup_dataset(df):
    """Converts pandas dataframe into xarray dataset."""
    ds = df.to_xarray()
    ds['height'].attrs['units'] = 'm'
    ds['temperature'].attrs['units'] = 'K'
    ds['pressure'].attrs['units'] = 'hPa'
    ds['relative_humidity'].attrs['units'] = 'percent'

    ds = ds.metpy.quantify()
    return ds

def build_layer_df(index_vector, data):
    """Selects the data at the top and bottom of layers of constant sign
    based on the sign vector and differences the variables in data across
    the layer. index_vector should be a list, and data should be an xarray
    dataset. Returns a pandas DataFrame."""
    
    idxb = numpy.array(index_vector[:-1])
    idxt = numpy.array(index_vector[1:])
    layer_dict = {}

    for cc in [cc for cc in data.variables if (cc != 'date')]:
        layer_dict[cc + '_base'] = data.sel(index=idxb)[cc].values
        layer_dict[cc + '_top'] = data.sel(index=idxt)[cc].values
        
    layer_df = pandas.DataFrame(layer_dict)
    
    if len(layer_df) == 0:
        layer_df = pandas.DataFrame(columns = layer_df.columns, index=[0])
    layer_df['date'] = numpy.array(data.date.values[0])    
    
    return layer_df


def merge_layers(index_list, layer_df, check_var, threshold, upper=True):
    """Scans through layer df. If check_var_diff is under threshold, drop the layer indices
    from index_list. If upper=false then drop layer indices if over threshold."""
    
    delta = layer_df[check_var + '_top'] - layer_df[check_var + '_base']
    remove_list = []
    for idx in layer_df.index.values[1:-1]:
        if upper==True:
            if delta.loc[idx] < threshold:
                remove_list.append(int(layer_df.loc[idx, 'index_base']))
                remove_list.append(int(layer_df.loc[idx, 'index_top']))
        elif upper==False:
             if delta.loc[idx] > threshold:
                remove_list.append(int(layer_df.loc[idx, 'index_base']))
                remove_list.append(int(layer_df.loc[idx, 'index_top'])) 
    remove_list = numpy.unique(remove_list)
        
    for idx in range(len(remove_list)):
        index_list.remove(remove_list[idx])
