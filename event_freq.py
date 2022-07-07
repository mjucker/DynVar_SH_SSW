import xarray as xr
from tabulate import tabulate
import pandas as pd
import numpt as np


seasons = {'JJASON':[6,11],'JJA':[6,8],'SON':[9,11]}

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

vxmoms = xr.open_mfdataset('vxmoms/ERA5_vxmoms_*_{0}hPa_{1}km.nc'.format(10,30.5))
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

## compute frequencies
def DetectMinMaxPeriods(ds,thresh,sep=20,period=7,time='time',kind='max'):
    '''
    Find events below (kind='min') or above (kind='max') a given threshold for a certain period of time. Events are merged into the earlier if less than a given separation time between them.

    INPUTS:
        ds:     xarray.dataarray used to detect events
        thresh: threshold to define events
        sep:    minimum separation of individual events (from end to start)
        period: minimum duration of each event, i.e. #days above threshold
        time:   name of time dimension
        kind:   find minimum if 'min', maximum if 'max'

    OUTPUTS:
        stats: xarray.Dataset of 
        duration:      duration of each event
        extreme_value: most extreme value during each event period.
        event_dates:   start and end dates for each event
    '''
    ds = ds.rename({time:'time'})
    da = ds.rolling(time=period)
    if kind.lower() == 'max':
        da = da.min()
        times = da > thresh
    elif kind.lower() == 'min':
        da = da.max()
        times = da < thresh
    event_all = da.isel(time=times)
    timestep = ds.time[1] - ds.time[0]
    e = 0
    event_id = [e]
    tm1 = event_all.time[0]
    for t in event_all.time[1:]:
        delta_t = t - tm1
        if delta_t > sep*timestep:
            e += 1
        tm1 = t
        event_id.append(e)
    ex = xr.DataArray(event_id,coords=[event_all.time],name='event_id')
    unique_events = np.unique(ex)
    duration = []
    extreme  = []
    se_dates = []
    for ue in unique_events:
        filtr = ex == ue
        prev_date = ex.isel(time=filtr).time[0].values-7*timestep
        end_date  = ex.isel(time=filtr).time[-1].values
        se_dates.append([prev_date.values,end_date])
        duration.append(period+sum(filtr.values)-1)
        if kind == 'min':
            extreme.append(ds.sel(time=slice(prev_date,end_date)).min().values)
        elif kind == 'max':
            extreme.append(ds.sel(time=slice(prev_date,end_date)).max().values)
    unicord = [('event',unique_events)]
    onx  = xr.DataArray(se_dates,coords=unicord+[('dates',['start','end'])],name='event_dates')
    durx = xr.DataArray(duration,coords=unicord,name='duration')
    extx = xr.DataArray(extreme,coords=unicord,name='extreme_value')
    outx = xr.merge([durx,extx,onx])
    outx.attrs['variable'] = ds.name
    options = {'max':'above','min':'below'}
    outx.attrs['method'] = 'individual {0}-day periods {1} {2}. Events are considered the same if spaced by less than {3} days'.format(period,options[kind],thresh,sep)
    return outx 


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

# we want to align everything so that similar dates are 
#  assigned to similar event ids
def FindUniqueEvents(events,event_sep):
    '''
      Returns a list of all events which happen within event_sep days of each other.
       This is across seasons, detection methods, and percentiles.

    INPUTS:
       events: xr.Dataset constructed as per above code.
       event_sep: time interval within which two events are considered the same.
    OUTPUTS:
       unique_events: list of onset dates of unique events
    '''
    all_events = []
    for season in events.season.values:
        for var in events.variable.values:
            for perc in events.percentile.values:
                all_events = all_events + [e for e in events.sel(season=season,variable=var,percentile=perc,dates='start').event_dates.values if np.isfinite(e)]
    all_events = np.unique(all_events)
    # we now have a list of unique event times
    # need to merge events which happen within event_sep days
    check_different = np.diff(all_events) > event_sep*np.timedelta64(1,'D')
    unique_events = all_events[[True]+list(check_different)]
    return unique_events

def AssignUniqueEvent(events,unique_events):
    '''
    Assigns given events to a list of unique events based on shortest distance.
    
    INPUTS:
      events: a list of events - these are time arrays
      unique_events: a list of unique events to which events are attributed to. also a list of time events
    OUTPUTS:
      indx: indices which map events to unique_events such that unique_events[indx] <=> events
    '''
    indx = []
    for event in events:
        if np.isfinite(event):
            diffs = unique_events - event
            indx.append(np.argmin(np.abs(diffs)))
    return np.array(indx)

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
            dates = events.sel(season=season,variable=var,percentile=perc,dates='start').event_dates.values
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

