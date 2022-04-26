"""Apply the inversion detection algorithm to the files in dataloc.
"""
import pandas as pd
import sys
import os
# sys.path.append('d:\\python_workspaces\\python workspace\\5-invclim\\invclim\\')
sys.path.append('./')
import invclim.core as icc
import invclim.invfinder as iif
import numpy as np

recalculate = False

# significant level inversions
dataloc = './Data/Soundings/'

params={'max_embed_depth': 100, 
        'min_dz': 0, # units('m'),
        'min_dp': 0, # units('hPa'),
        'min_dt': 0, # units('K'),
        'min_drh': 0, # units('percent'),
        'rh_or_dt': False}

def find_inversions(group):
    """Function to apply to each sounding for groupby('date').apply
    Includes extra function to merge inversion layers since my code still
    doesn't do that properly.
    """
    
    def check_interstitial_thickness(inv_df, max_embed_depth = 100):
        """Merge embedded negative lapse rate layers. If there is more than
        one inversion, check the interstitial thickness. If any are less than
        max_embed_depth, replace the lowest inversion with the merged inversion, 
        drop the row with the h"""
    
        if len(inv_df) > 1:
            interstitial = []
            for idx in inv_df.index[1:]:
                interstitial.append(inv_df.loc[idx, 'height_base'] - inv_df.loc[idx-1, 'height_top'])
            interstitial = np.array(interstitial)
            while np.any(interstitial < max_embed_depth):

                idx = 0
                for idx in range(len(interstitial)):
                    if interstitial[idx] > max_embed_depth:
                        idx += 1
                    else:
                        break

                for col in inv_df.columns:
                    if len(col.split('_')) > 1:
                        if col.split('_')[1] == 'top':
                            inv_df.loc[inv_df.index[idx], col] = inv_df.loc[inv_df.index[idx+1], col]

                inv_df.drop(inv_df.index[idx+1], inplace=True)
                inv_df['inv_number'] = np.arange(1, len(inv_df) + 1)

                if len(inv_df) > 1:
                    interstitial = []
                    for idx in range(1, len(inv_df)-1):
                        interstitial.append(inv_df.loc[
                            inv_df.index[idx], 'height_base'] - inv_df.loc[inv_df.index[idx-1], 'height_top'])

                    interstitial = np.array(interstitial)
                else:
                    break
        return inv_df
    
    df = icc.setup_dataset(group.loc[:, ['date','pressure','height','temperature','relative_humidity']
                                    ].reset_index(drop=True)) # put the reset into setup_dataset

    inv = iif.invfinder(df, params)
    inv.index.names = ['inv_number']
    inv.reset_index(inplace=True)
    inv['date'] = group.name

    # invfinder still has a merge layers issue, i.e., it doesn't catch when the 
    # negative lapse rate should get skipped! This applies the final merge step.
    inv = check_interstitial_thickness(inv, params['max_embed_depth'])
    
    return inv

arctic_stations = pd.read_csv('./Data/arctic_stations_long.csv').set_index("station_id")
saveloc = './Data/Inversions/'

# Find out which stations have already had inversions calculated
calculated = os.listdir('./Data/Inversions/')
calculated = [x for x in calculated if len(x) < 30]
calculated = [x.split('_')[0] for x in calculated if x[-3:] == 'csv']

if recalculate:
    calculate = [site for site in arctic_stations.index]

else:
    calculate = []
    for site in arctic_stations.index:
        if site in calculated:
            print('Already calculated', site)
        else:
            calculate.append(site)

for site in calculate: 
    df = pd.read_csv(dataloc + site + '-cleaned-soundings.csv')
    print(len(df))
    df['date'] = pd.to_datetime(df.date.values)
    elev = max(0, df.height.min())
    df = df.loc[df.height < elev + 5000]
    
    try:
        inv = df.groupby('date').apply(find_inversions)
        inv.to_csv(saveloc + site + '_inversions.csv')
        del inv
    except:
        print(site + ' find inversions failed')