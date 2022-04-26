### Plot the locations of the weather stations, along with environmental information
import xarray as xr
import pandas as pd
import proplot as pplt
import numpy as np

arctic_stations = pd.read_csv('../Data/arctic_stations_long.csv')
arctic_stations.set_index('station_id', inplace=True)

sic = xr.open_dataset('../../../Data/goddard_bt_sic_monthly.nc')
sic_seasonal = sic.groupby('time.season').mean()

proj = pplt.Proj('nplaea')
fig, axs = pplt.subplots(ncols=1, proj=proj, height=6, width=10)
axs.format(
    land=True,
    landcolor='k',
    rivers=True,
    riverscolor='indigo2',
    ocean=True,# no gridlines poleward of 80 degrees
    boundinglat=60,
    oceancolor='indigo2',
    suptitle='Locations of weather stations'
)

arctic_stations.loc['ICM00004089', 'region'] = 'Maritime'
colors = {letter: color['color'] for letter, color in zip(['Eastern Eurasia',
                                                                  'Eastern North America',
                                                                  'Greenland',
                                                                  'Maritime',
                                                                  'Western Eurasia',
                                                                  'Western North America'],
                                                pplt.Cycle('538',9))}


# Include only if we need the elevation overlay
#axs.contourf(ele.lon, ele.lat, ele, cmap='oleron', vmin=-1500, vmax=1500, levels=40)#vmin=0, vmax=300, levels=30)

for i, site in enumerate(arctic_stations.index):
    name = arctic_stations.loc[site,'name']
    lat = arctic_stations.loc[site, 'lat']
    lon = arctic_stations.loc[site, 'lon']
    region = arctic_stations.loc[site, 'region']

    if pd.to_datetime(arctic_stations.loc[site, 'begin_date']) != pd.to_datetime('2000-01-01 00:00'):
        asterisk = '*'
    else:
        asterisk = ''
        
    axs.plot(lon, lat, marker='o', linewidth=0, color=colors[region],
        label=str(i + 1) + ' ' + name + asterisk, legend='r', legend_kw={'ncols':2, 'order':'F'})

    axs.text(x=lon, 
             y=lat,
             text=str(i+1),
             verticalalignment='center',
             color='white',
             fontsize=10,
             bbox={'facecolor':colors[region], 'boxstyle': 'circle'})
    lats = sic_seasonal.lat
    lons = sic_seasonal.lon
    for season, color in zip(['MAM', 'SON'], 
                             ['lightgray', 'white', 'gray', 'darkgray']):
        X = sic_seasonal.sel(season=season)['goddard_bt_seaice_conc_monthly']
        X[X.lat > 87, :] = 1
        X = np.ma.masked_less(X, 0.15)
        axs.contourf(lons, lats, X, levels=[0, 0.15, 1],
                    color=color, zorder=2, globe=True, alpha=1)
fig.save('./Images/Paper/station_locations.pdf')