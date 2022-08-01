import xarray as xr
from tabulate import tabulate
import pandas as pd
import numpy as np
import seaborn as sns
import os
from DynVar_SH_SSW.functions import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l',dest='level',required=True,type=float,help='Extract this pressure level.')
parser.add_argument('-r',dest='roll',required=True,type=int,help='Number of days beyond threshold.')
parser.add_argument('-q',dest='quants',default=[0.01,0.05,0.10,0.90,0.95,0.99],nargs='+',type=float,help='Quantiles to detect.')
args = parser.parse_args()



seasons = {'JJASON':[6,11],'JJA':[6,8],'SON':[9,11]}
# only need to write onset dates of longest season
write_season = 'JJASON'

quants = args.quants

level = args.level

roll = args.roll

event_sep = 20

sam = xr.open_dataarray('zpc_sam/zpc_sam.nc').sel(pres=level)
sam.load()

percentiles = {}
for season,months in seasons.items():
    filtr = (sam.time.dt.month >= months[0])*(sam.time.dt.month <= months[1])
    ds = sam.isel(time=filtr)
    percentiles[season] = {}
    for thresh in quants:
        quantile = ds.quantile(thresh).values
        percentiles[season][thresh] = float(quantile)

# print the table
nseasons = len(seasons.keys())
nquantiles = len(quants)
title_row = '\\begin{table}[]\n    \centering\n    \\begin{tabular}{l||'+''.join(['r|r||']*nseasons)+'}\n'
season_row = ''.join([' & \multicolumn{'+str(nquantiles)+'}{|c||}{'+str(s)+'}' for s in seasons.keys()]) + ' \\\ \n'
quants_row = ' & '+' & '.join(['{0:2.0%} '.format(q) for q in quants]*nseasons).replace('%','\%') + ' \\\ \n'
lines = ''
for season in seasons:
    for q in quants:
        lines = lines+' & {0:5.2f}'.format(percentiles[season][q])
lines = lines + ' \\\ \n '
foot_row = '    \end{tabular}\n    \caption{SAM (polar cap) percentile threshold values for different seasons.}\n    \label{tab:quants}\n\end{table}\n'

# put everything together:
table_string = title_row+season_row+quants_row+'\hline \n'+lines+'\hline \n'+foot_row
print('HERE ARE THE PERCENTILE VALUES FOR CENTROID LATITUDE AND ASPECT RATIO:')
print(table_string)


events = []
for season,months in seasons.items():
    filtr = (sam.time.dt.month >= months[0])*(sam.time.dt.month <= months[1])
    ds = sam.isel(time=filtr)
    vstats = []
    for perc in quants:
            stats = DetectMinMaxPeriods(ds,percentiles[season][perc],sep=event_sep,period=roll,kind='auto')
            if stats is None:
                continue
            stats['percentile'] = perc
            stats['thresh'] = percentiles[season][perc]
            vstats.append(stats)
            # write the CSV file
            if season == write_season:
                init_txt = '# SAM defined as standardize polar cap (60-90) geopotential height at {0} hPa'.format(level)+os.linesep
                init_txt += '# '+stats.attrs['method']+os.linesep
                WriteCSV(stats.onset_date,init_txt,'csv/onset_dates_sam_r{0}_{1}_{2}hPa_q{3}.csv'.format(roll,season,level,perc))
    vstats = xr.concat(vstats,dim='percentile')
    vstats['season'] = season
    events.append(vstats)
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
    for perc in events.percentile.values:
        head = '\n'.join([season,str(perc)])
        dates = events.sel(season=season,percentile=perc).onset_date.values
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

# create rolling decade statistic on the number of events
#  this allows for error bars
dec_stat = []
for yr in range(1979,2013):
    filtr = (events.onset_date.dt.year >= yr)*(events.onset_date.dt.year < yr+10)
    reduced = events.where(filtr).reduce(np.isfinite).onset_date.sum('event')
    reduced = reduced.assign_coords({'season':events.season,'percentile':events.percentile})
    reduced['decade'] = str(yr)+'-'+str(yr+9)
    dec_stat.append(reduced)
dec_stat = xr.concat(dec_stat,dim='decade')
dec_stat.name = 'events per decade'
# now convert this into individual columns for a DataFrame
dec_stat_cols = {}
for season in dec_stat['season'].values:
    for perc in dec_stat['percentile'].values:
        head = '\n'.join([season,str(perc)])
        dec_stat_cols[head] = dec_stat.sel(season=season,percentile=perc).values
dec_stat_cols = pd.DataFrame(dec_stat_cols)

# plot the stats
colors = sns.color_palette()
hatches = ['','//','--','\\\\','||','..','oo']*3
#hatches = ["*", "/", "o", "x"]
decors = {'sign':{},'perc':{}}
s=0
pperc=0
for c,perc in enumerate(dec_stat['percentile'].values):
    if pperc*(perc-0.5) < 0:
       s += 1
    decors['sign'][str(perc)] = colors[s]
    pperc = perc-0.5
    decors['perc'][str(perc)] = hatches[c]
g = sns.barplot(data=dec_stat_cols)
for l,label in enumerate(g.axes.get_xticklabels()):
    txt = label.get_text()
    season,perc = txt.split('\n')
    g.patches[l].set_facecolor(decors['sign'][perc])
    g.patches[l].set_hatch(decors['perc'][perc])
ax = g.axes
fig= g.figure
sns.despine(ax=ax,offset=10)
ax.set_title('event frequency [#/decade]')
ax.set_ylabel('# events per decade')
fig.set_figwidth(fig.get_figwidth()*1.4)
outFile = 'figures/sam_r{0}_{1}hPa_events_per_decade.pdf'.format(roll,level)
fig.savefig(outFile,transparent=True,bbox_inches='tight')
print(outFile)
