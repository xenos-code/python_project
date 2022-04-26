"""Calculates the number of soundings and the number of pressure levels available
for each month and plots the result."""

import pandas as pd
import proplot as pplt
import numpy as np

arctic_stations = pd.read_csv('./Data/arctic_stations_long.csv')
arctic_stations.set_index('station_id', inplace=True)

soundings = {}
for site in arctic_stations.index:
    try:
        soundings[site] = pd.read_csv('./Data/Soundings/' + site + '-cleaned-soundings.csv')
        soundings[site]['date'] = pd.to_datetime(soundings[site].date.values)        
    except:
        print('Missing sounding data for ' + site)

pressure_resolution = {}
sounding_count = {}
sounding_count12 = {}
sounding_count00 = {}
for site in soundings:
    daily = soundings[site].groupby('date').count().pressure
    pressure_resolution[site] = daily.resample('1MS').mean()
    sounding_count[site] = daily.resample('1MS').count()
    sounding_count00[site] = daily.loc[(daily.index.hour > 22) | (daily.index.hour < 2)].resample('1MS').count()
    sounding_count12[site] = daily.loc[(daily.index.hour > 10) & (daily.index.hour < 14)].resample('1MS').count()
    
    
    
counts00 = pd.DataFrame({site: sounding_count00[site] for site in sounding_count})
counts12 = pd.DataFrame({site: sounding_count12[site] for site in sounding_count})
countsany = pd.DataFrame({site: sounding_count[site] for site in sounding_count})


ps_timeseries = pd.DataFrame(pressure_resolution).resample('1MS').mean()

#arctic_stations = arctic_stations.loc[arctic_stations.n_missing_months_post_2006 <= 1]
fig, ax = pplt.subplots(nrows=3, ncols=2, share=False)
colors = {letter: color['color'] for letter, color in zip(np.unique(arctic_stations.region),
                                                pplt.Cycle('538',8))}
for ax, site in zip(np.ravel(ax)[:-1], arctic_stations.index):
    region = arctic_stations.loc[site, 'region']
    nn = sounding_count00[site]
    ax.plot(nn, marker='o', linestyle='', color='k')
    ax.format(ylabel='Number of soundings per month',
              title=arctic_stations.loc[site, 'name'],
              xlim=(pd.to_datetime('1990-01-01'), pd.to_datetime('2020-01-01')))
    nn = sounding_count12[site]
    ax.plot(nn, marker='.', linestyle='', color=colors[region])
    ax.format(ylabel='n soundings', ylim=(0,35))
    
    if site[0] == 'R':
        ax.axvline(pd.to_datetime('2005-01-01'), color='dark gray')
    ax2 = ax.twinx()
    ax2.format(ylabel='n levels', ylim=(0,35))
    ax2.plot(ps_timeseries.loc[:,site].resample('1YS').mean(),
             marker='.', linestyle='-', color='gray')
ax.format(suptitle='Number of soundings in each month for 00Z (black) and 12Z (colored) and average number of levels below 500 hPa', xlabel='')
fig.save('./Images/Supplement/nsoundings_by_month.pdf')
pplt.close(fig)