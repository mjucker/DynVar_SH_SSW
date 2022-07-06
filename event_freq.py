import xarray as xr
from tabulate import tabulate


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
            stats = DetectMinMaxPeriods(ds['_'.join(var.split(' '))],percentiles[season][var][perc],sep=20,period=7,kind=minmax[var])
            stats['percentile'] = perc
            vstats.append(stats)
        vstats = xr.concat(vstats,dim='percentile')
        vstats['variable'] = var
        sstats.append(vstats)
    sstats = xr.concat(sstats,dim='variable')
    sstats['season'] = season
    events.append(sstats)
events = xr.concat(events,dim='season')

