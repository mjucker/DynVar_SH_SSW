import xarray as xr
import os
import numpy as np



def DetectMinMaxPeriods(ds,thresh,sep=20,period=7,time='time',kind='auto'):
    '''
    Find events below (kind='min') or above (kind='max') a given threshold for a certain period of time. Events are merged into the earlier if less than a given separation time between them.

    INPUTS:
        ds:     xarray.dataarray used to detect events
        thresh: threshold to define events
        sep:    minimum separation of individual events (from end to start)
        period: minimum duration of each event, i.e. #days above threshold
        time:   name of time dimension
        kind:   find minimum if 'min', maximum if 'max'.
                if 'auto': minimum if thresh <= 0, maximum elsewhise

    OUTPUTS:
        stats: xarray.Dataset of 
          duration:      duration of each event
          extreme_value: most extreme value during each event period.
          event_dates:   start and end dates for each event
    '''
    ds = ds.rename({time:'time'})
    da = ds.rolling(time=period)
    if kind == 'auto':
        if thresh <= 0:
            kind = 'min'
        else:
            kind = 'max'
    if kind.lower() == 'max':
        da = da.min()
        times = da > thresh
    elif kind.lower() == 'min':
        da = da.max()
        times = da < thresh
    # in case no event has been detected
    if np.sum(times) == 0:
        return None
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
    start_dates = []
    end_dates = []
    for ue in unique_events:
        filtr = ex == ue
        prev_date = ex.isel(time=filtr).time[0].values-7*timestep
        end_date  = ex.isel(time=filtr).time[-1].values
        start_dates.append(prev_date.values)
        end_dates.append(end_date)
        duration.append(period+sum(filtr.values)-1)
        if kind == 'min':
            extreme.append(ds.sel(time=slice(prev_date,end_date)).min().values)
        elif kind == 'max':
            extreme.append(ds.sel(time=slice(prev_date,end_date)).max().values)
    unicord = [('event',unique_events)]
    onx  = xr.DataArray(start_dates,coords=unicord,name='onset_date')
    edx  = xr.DataArray(end_dates,coords=unicord,name='end_date')
    durx = xr.DataArray(duration,coords=unicord,name='duration')
    extx = xr.DataArray(extreme,coords=unicord,name='extreme_value')
    outx = xr.merge([durx,extx,onx,edx])
    outx.attrs['variable'] = ds.name
    options = {'max':'above','min':'below'}
    outx.attrs['method'] = 'individual {0}-day periods of {4} {1} {2}. Events are considered the same if spaced by less than {3} days'.format(period,options[kind],thresh,sep,ds.name)
    return outx 


def WriteCSV(onset_dates,init_text,filename):
    '''
    Write CSV files with all onset dates. Will write each date on one line following the format year,month,day
    INPUTS:
       onset_dates:  xarrat.DataArray of type time. Defines the onset dates to be written as lines
       init_text:    any file header to be written before the onset dates
       filename:     name of file to be created
    '''
    import os
    with open(filename,'w') as csvfile:
        csvfile.write(init_text)
        for event in onset_dates:
            csvfile.writelines(str(event.dt.strftime('%Y,%m,%d').values)+os.linesep)
    print(filename)

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
        seldict = {'season':season}
        for perc in events.percentile.values:
            seldict['percentile']=perc
            if 'variable' in events:
                for var in events.variable.values:
                    seldict['variable'] = var
                    all_events = all_events + [e for e in events.sel(**seldict).onset_date.values if np.isfinite(e)]
            else:
                all_events = all_events + [e for e in events.sel(**seldict).onset_date.values if np.isfinite(e)]
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
