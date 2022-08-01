import xarray as xr
from tabulate import tabulate
import pandas as pd
import numpy as np
import seaborn as sns
import os
from DynVar_SH_SSW.functions import *


seasons = {'JJASON':[6,11],'JJA':[6,8],'SON':[9,11]}
# only need to write onset dates of longest season
write_season = 'JJASON'

quants = [0.10,0.05]

invert_quants = {'centroid latitude': False,
                 'aspect ratio'     : True
}

minmax = {'centroid latitude': 'min',
          'aspect ratio'     : 'max',
}

level = 10
edge  = 30.5

roll = 7

event_sep = 20

vxmoms = xr.open_mfdataset('vxmoms/ERA5_vxmoms_*_{0}hPa_{1}km.nc'.format(level,edge))
vxmoms.load()

percentiles = {}
for season,months in seasons.items():
    filtr = (vxmoms.time.dt.month >= months[0])*(vxmoms.time.dt.month <= months[1])
    ds = vxmoms.isel(time=filtr)
    percentiles[season] = {}
    for i,items in enumerate(invert_quants.items()):
        var = items[0]
        percentiles[season][var] = {}
        for q in quants:
            if items[1]:
                thresh = 1-q
            else:
                thresh = q
            quantile = ds['_'.join(var.split(' '))].quantile(thresh).values
            percentiles[season][var][q] = float(quantile)

# print the table
nseasons = len(seasons.keys())
nquantiles = len(quants)
title_row = '\\begin{table}[]\n    \centering\n    \\begin{tabular}{l||'+''.join(['r|r||']*nseasons)+'}\n'
season_row = ''.join([' & \multicolumn{'+str(nquantiles)+'}{|c||}{'+str(s)+'}' for s in seasons.keys()]) + ' \\\ \n'
quants_row = ' & '+' & '.join(['{0:2.0%} '.format(q) for q in quants]*nseasons).replace('%','\%') + ' \\\ \n'
lines = ''
for var in invert_quants.keys():
    lines = lines + var
    for season in seasons:
        for q in quants:
            lines = lines+' & {0:5.2f}'.format(percentiles[season][var][q])
    lines = lines + ' \\\ \n '
foot_row = '    \end{tabular}\n    \caption{Aspect ratio and centroid latitude percentile threshold values for different seasons. For each season, the most extreme 10\% and 5\% values are shown, corresponding to the 90th and 95th percentiles for aspect ratio, and the 10th and 5th percentiles for centroid latitude.}\n    \label{tab:quants}\n\end{table}\n'

# put everything together:
table_string = title_row+season_row+quants_row+'\hline \n'+lines+'\hline \n'+foot_row
print('HERE ARE THE PERCENTILE VALUES FOR CENTROID LATITUDE AND ASPECT RATIO:')
print(table_string)


events = []
for season,months in seasons.items():
    filtr = (vxmoms.time.dt.month >= months[0])*(vxmoms.time.dt.month <= months[1])
    ds = vxmoms.isel(time=filtr)
    sstats = []
    for var in invert_quants.keys():
        vstats = []
        for perc in quants:
            stats = DetectMinMaxPeriods(ds['_'.join(var.split(' '))],percentiles[season][var][perc],sep=event_sep,period=roll,kind=minmax[var])
            stats['percentile'] = perc
            vstats.append(stats)
            # write the CSV file
            if season == write_season:
                init_txt = '# Vortex moment definition: geopotential height at {0} hPa, vortex edge = {1} km'.format(level,edge)+os.linesep
                init_txt += '# '+stats.attrs['method']+os.linesep
                WriteCSV(stats.onset_date,init_txt,'csv/onset_dates_vxmoms_{0}_{1}_{2}hPa_{3}km_q{4}.csv'.format('_'.join(var.split(' ')),season,level,edge,perc))
        vstats = xr.concat(vstats,dim='percentile')
        vstats['variable'] = var
        sstats.append(vstats)
    sstats = xr.concat(sstats,dim='variable')
    sstats['season'] = season
    events.append(sstats)
events = xr.concat(events,dim='season')
del events.attrs['variable']
events.attrs['method'] = 'individual {0}-day periods below/above given percentiles. Events are considered the same if spaced by less than {1} days'.format(roll,event_sep)

## Now get some statistics
# get the onset dates for each season, definition, and threshold


unique_events = FindUniqueEvents(events,event_sep)
nevents = len(unique_events)

# then, assign an event id to each individual event
s_dates = []
for season in events.season.values:
    date_cols = []
    for var in events.variable.values:
        short_var = ''.join([c[0] for c in var.split()]).upper()
        for perc in events.percentile.values:
            head = '\n'.join([season,short_var,str(perc)])
            dates = events.sel(season=season,variable=var,percentile=perc).onset_date.values
            event_ids = AssignUniqueEvent(dates,unique_events)
            fins = np.isfinite(dates)
            dates = pd.to_datetime(dates[fins]).strftime('%Y-%b-%d')
            dates_v = ['-']*nevents
            for d,e in enumerate(event_ids):
                dates_v[e] = dates[d]
            dx = xr.DataArray(dates_v,coords=[('event',np.arange(nevents))],name=head)
            date_cols.append(dx)
    season_dates = xr.merge(date_cols)
    # now print this as a latex table as well
    print(tabulate(season_dates.to_dataframe(),headers='keys',showindex=False,tablefmt='latex_longtable'))
    print('\clearpage')
    s_dates.append(season_dates)
all_dates = xr.merge(s_dates)

##
# Next, we want to construct histograms with number of events by method
#def SumDates(da,axis=None):
#    ln = len(da)
#    if ln == 0:
#        return 0
#    empty = np.sum(da == '-')
#    return ln-empty
#
#def CreateSumArrays(ds):
#    sums = ds.reduce(SumDates)
#    dims = []
#    vals = []
#    for var in sums.data_vars:
#        dims.append(var)
#        vals.append(sums[var].values)
#    return xr.DataArray(vals,coords=[('data',dims)],name='totals')

# create rolling decade statistic on the number of events
#  this allows for error bars
dec_stat = []
for yr in range(1979,2013):
    filtr = (events.onset_date.dt.year >= yr)*(events.onset_date.dt.year < yr+10)
    reduced = events.where(filtr).reduce(np.isfinite).onset_date.sum('event')
    reduced = reduced.assign_coords({'season':events.season,'variable':events.variable,'percentile':events.percentile})
    reduced['decade'] = str(yr)+'-'+str(yr+9)
    dec_stat.append(reduced)
dec_stat = xr.concat(dec_stat,dim='decade')
dec_stat.name = 'events per decade'
# now convert this into individual columns for a DataFrame
dec_stat_cols = {}
for season in dec_stat['season'].values:
    for var in dec_stat['variable'].values:
        short_var = ''.join([c[0] for c in var.split()]).upper()
        for perc in dec_stat['percentile'].values:
            head = '\n'.join([season,short_var,str(perc)])
            dec_stat_cols[head] = dec_stat.sel(season=season,variable=var,percentile=perc).values
dec_stat_cols = pd.DataFrame(dec_stat_cols)

# plot the stats
colors = sns.color_palette()
hatches = ['','//','--']*3
#hatches = ["*", "/", "o", "x"]
decors = {'var':{},'perc':{}}
for v,var in enumerate(dec_stat['variable'].values):
    short_var = ''.join([c[0] for c in var.split()]).upper()
    decors['var'][short_var] = colors[v]
for c,perc in enumerate(dec_stat['percentile'].values):
    decors['perc'][str(perc)] = hatches[c]
g = sns.barplot(data=dec_stat_cols)
for l,label in enumerate(g.axes.get_xticklabels()):
    txt = label.get_text()
    season,var,perc = txt.split('\n')
    g.patches[l].set_facecolor(decors['var'][var])
    g.patches[l].set_hatch(decors['perc'][perc])
ax = g.axes
fig= g.figure
sns.despine(ax=ax,offset=10)
ax.set_title('event frequency [#/decade]')
ax.set_ylabel('# events per decade')
fig.set_figwidth(fig.get_figwidth()*1.4)
outFile = 'figures/events_per_decade.pdf'
fig.savefig(outFile,transparent=True,bbox_inches='tight')
print(outFile)
