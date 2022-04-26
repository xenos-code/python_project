"""Calculate inversion frequency by height and plot the result."""
import numpy as np
import pandas as pd
import proplot as pplt
pplt.rc.update({'suptitle.size': 12, 'title.size': 8})

arctic_stations = pd.read_csv('./Data/arctic_stations_long.csv')
arctic_stations.set_index('station_id', inplace=True)
# arctic_stations = arctic_stations.loc[(arctic_stations.n_missing_months_post_2005 <= 1) | 
#                                       (arctic_stations.index == 'GLM00004417')]

start_date = pd.to_datetime('2000-01-01 00:00')
end_date = pd.to_datetime('2019-12-31 23:00')

inversions = {}
for site in arctic_stations.index:
    start_date = arctic_stations.loc[site, 'begin_date']
    end_date = arctic_stations.loc[site, 'end_date']
    inversions[site] = pd.read_csv('./Data/Inversions/' + site + '_inversions.csv')
    inversions[site]['date'] = pd.to_datetime(inversions[site].date.values)
    inversions[site] = inversions[site].loc[(inversions[site].date >= start_date) &
                                            (inversions[site].date <= end_date)]
def inv_indicator(inv_df, zgrid):
    """Returns a dataframe dimensions (n_obvs x n_heights) with 
    entries 1 if inversion is present at that height and 0 otherwise."""
    ind_list = []
    names = []
    for name, group in inv_df.groupby('date'):

        ind = zgrid*0
        for height_base, height_top in zip(group.height_base, group.height_top):
            ind += np.array((height_base <= zgrid) & (height_top > zgrid))
        ind_list.append(ind)
        names.append(name)
    return pd.DataFrame(np.vstack(ind_list), index=names, columns=zgrid)        

def get_phi(indicator_df):
    """From the indicator df output by inv_indicator, compute lag 1 autocorrelation
    via the Pearson method for each month and each height."""
    corr_coef = np.zeros((12, len(zgrid)))
    for ii, month in enumerate(np.arange(1, 13)):
        for jj, col in enumerate(indicator_df.columns):
            corr_coef[ii, jj] = indicator_df.loc[
                indicator_df.index.month == month, col].shift(1).corr(
                indicator_df.loc[
                    indicator_df.index.month == month, col], method='pearson')   
    return pd.DataFrame(
        data=corr_coef, index=np.arange(1, 13), 
        columns=indicator_df.columns)

def standard_error_adj(f, phi):
    """Computes the error in the proportion adjusted for autocorrelation phi.
    f should be a dataframe or array with nrows = nobservations, ncols=nheights,
    and phi is an array with len nheights."""
    n = len(f)
    v =  (1+phi)/(1-phi)
    v[v < 1] = 1
    se = np.sqrt(f.mean(axis=0)*(1-f.mean(axis=0))/n *v)
    return se


def season(m):
    if m in [12, 1, 2]:
        return 'DJF'
    elif m in [3,4,5]:
        return 'MAM'
    elif m in [6,7,8]:
        return 'JJA'
    else:
        return 'SON'
    
    
freqs = {}
phis = {}
errs = {}
n = {}

for site in arctic_stations.index:
    # setting first point at 5m AGL so changes in elevation aren't as important.
    # could adjust actual time series tho.
    zgrid = arctic_stations.loc[site, 'elevation'] + 5 + np.arange(25, 3000, 50)
    
    # flags 1 if an inversion overlaps that height and 0 if not
    indicator_df = inv_indicator(inversions[site], zgrid)
    
    # monthly average of indicators is the frequency
    freqs[site] = indicator_df.resample('1MS').mean()
    
    # estimate the correlation at each height with pearson correlation coefficient
    phis[site] = get_phi(indicator_df)
    
    # adjusted standard error for proportion if phi is positive, otherwise use standard error
    errs[site] = standard_error_adj(freqs[site], phis[site])
    
    # number of observations in each month
    n[site] = indicator_df.resample('1MS').count().iloc[:,1]
    
# plot the seasonal inversion frequency plots with shading
colors = {letter: color['color'] for letter, color in zip(['DJF', 'MAM', 'JJA', 'SON'],
                                                pplt.Cycle('538', 4))}


# fig, axs = pplt.subplots(nrows=9, ncols=5, journal='ams4', wspace=0.1, hspace=0.2)
# for site, ax in zip(arctic_stations.index, np.ravel(axs)):
#     seas = np.array([season(m) for m in freqs[site].index.month])
#     for sn in ['JJA', 'DJF']:
#         z = freqs[site].columns - arctic_stations.loc[site, 'elevation']
#         shade=freqs[site].loc[seas==sn].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
#         ax.fill_betweenx(z, shade.loc[0.1,:].values, shade.loc[0.9,:].values,alpha=0.2, color=colors[sn])
#         ax.fill_betweenx(z, shade.loc[0.25,:].values, shade.loc[0.75,:].values,alpha=0.2, color=colors[sn])
#         ax.plot(shade.loc[0.5,:].values, z,  color=colors[sn], linestyle=':')
#         f = freqs[site].loc[seas==sn].mean(axis=0)
#         err = errs[site].loc[[12,1,2],:].mean(axis=0)
#         ax.plot(f.values, z, color=colors[sn], linewidth=1)
#         ax.plot(f.values - err.values, z, color=colors[sn], linewidth=1, linestyle='--')
#         ax.plot(f.values + err.values, z, color=colors[sn], linewidth=1, linestyle='--')
#         ax.format(urtitle=arctic_stations.loc[site, 'name'])
        
# axs.format(xreverse=False, xlim=(-0.1,1.1), 
#            ylabel='Altitude (m)', xlabel='Frequency', xtickminor=False, ytickminor=False, 
#            xlocator=[0, 0.25, 0.5, 0.75, 1], xticklabels=['0', '', '0.5', '', '1'],
#            ylocator=np.arange(0,3001,500), yticklabels=['', '1000', '', '2000', '', '3000'], 
#            xticklen=0, yticklen=0, ylim=(0,3100), abc=True, abcloc='lr')  

# #axs[-1,-1].axis('off')
# fig.save('../Images/Paper/seasonal_inv_freq.pdf')


### Plot all seasons, with and without secondary inversion ####
arctic_stations.loc['CHM00050774', 'region'] = 'CHINA'
regcolors = {letter: color['color'] for letter, color in zip(['Eastern Eurasia',
                                                                  'Eastern North America',
                                                                  'Greenland',
                                                                  'Maritime',
                                                                  'Western Eurasia',
                                                                  'Western North America',
                                                                  'CHINA'],
                                                pplt.Cycle('538',9))}
# fig, axs = pplt.subplots(nrows=6, ncols=1, journal='ams4', wspace=0.1, hspace=0.2)
fig, axs = pplt.subplots(nrows=5, ncols=1, journal='ams4', axwidth=2,share=False)
arctic_stations['idx'] = np.arange(1, len(arctic_stations)+1)
season_plot = ['DJF', 'MAM', 'JJA', 'SON'] #JJA夏季 DJF冬季 SON秋季 MAM春季
# season_label = ['冬', '春', '夏', '秋'] #JJA夏季 DJF冬季 SON秋季 MAM春季
for site, ax in zip(arctic_stations.index, np.ravel(axs)[0:len(arctic_stations.index)]):
    seas = np.array([season(m) for m in freqs[site].index.month])
    z = freqs[site].columns - arctic_stations.loc[site, 'elevation']
    
    for seasons in season_plot:
        f = freqs[site].loc[seas==seasons].mean(axis=0)
        nn = n[site]
        err = errs[site].loc[[12,1,2],:].mean(axis=0)
        ax.plot(f.values, z, color=colors[seasons], linewidth=1)
        ax.fill_betweenx(z, f.values - err.values, f.values + err.values, color=colors[seasons], 
                         alpha=0.5)
    
    handles = []
    for seasons in season_plot:
        handles.append(ax.plot([], [], label=seasons, color=colors[seasons]))
        
        
    #ax.format(urtitle=arctic_stations.loc[site, 'name'])

    region = arctic_stations.loc[site, 'region']

    ax.text(x=0.75, 
    y=2500,
    text=str(arctic_stations.loc[site, 'idx']),
    verticalalignment='center',
    color='white',
    fontsize=10,
    bbox={'facecolor':regcolors[region], 'boxstyle': 'circle'})
    
axs.format(xreverse=False, xlim=(-0.1,1.1),
           ylabel='Altitude-海拔高度 (m)', xlabel='Frequency-频次', abc=False, xtickminor=False, ytickminor=False, 
           xlocator=[0, 0.25, 0.5, 0.75, 1], xticklabels=['0', '', '0.5', '', '1'],
           ylocator=np.arange(0,3001,500), yticklabels=['', '500', '', '1500', '', '2500'], 
           xticklen=0, yticklen=0, ylim=(0,3100))
# for idx in np.arange(1, 8):
#     axs[-1, idx].axis('off')
#axs[-1,-1].legend(handles, labels=season_plot, ncols=1)
fig.legend(handles, labels=season_plot, ncols=1, loc='r')
fig.save('./Images/freq_by_height_seasons.pdf')

freqs_df = pd.concat(freqs)
freqs_df.to_csv('./Data/freq_by_height.csv')
phis_df = pd.concat(phis)
phis_df.to_csv('./Data/phis_by_height.csv')
pd.concat(n).to_csv('./Data/n_freqs.csv')
err.to_csv('./Data/err.csv')
