import datetime
import warnings

import geopandas as gpd
import pandas as pd


# DATAFRAME MANIPULATIONS AND TRANSFORMATIONS ---------------------------------
# -----------------------------------------------------------------------------
def _assemble_operational_data(feed):
    with warnings.catch_warnings():
        # This concerns this warnings:
        # https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        trips_and_routes = pd.merge(feed.trips,
                                    feed.routes,
                                    how='left',)
        
        op_data = (feed
                   .stop_times
                   .merge(trips_and_routes, how='left')
                   .merge(feed.stops, how='left')
                  )
    
    
    return op_data


def _agg_trips(trip_data, agg_on, col_with_intervals):
    """Aggregate trips by time interval, direction (whether Inbound
    or Outbound), and by a third attribute that is either "route_id"
    or "stop_id".
    
    Parameters
    -----------
    trip_data: DataFrame
        Operational data relating departue times with data on
        routes and stops
    agg_on: str {"route_id", "stop_id"}
        Defines wheter funtion will output trips by route or by stop
        
        TO DO: other string options will provide outputs, if they are
        valid columns, of course. Think if any of those might have meaning
        
    col_with_intervals: str {"hour", "window"}
        Name of column containing time intervals to aggregate under
        
    Returns
    --------
        agg_trips: DataFrame
    """
    trip_aggregation = (trip_data
                        .pivot_table('trip_id',
                                     index=[agg_on,
                                            'direction_id',
                                            col_with_intervals],
                                     aggfunc='count')
                        .reset_index()
                        .rename(columns={'trip_id': 'trips'})
                       )

    
    return trip_aggregation


# TIME RELATED WRANGLINGS -----------------------------------------------------
# -----------------------------------------------------------------------------
def _fix_departure_times(departures):
    """Some departures correspond to trips that initiated the day
    before, hence they are recorded as, e.g., 25:45 or 24:35, which
    might have its uses.
    
    This function is aimed at cases in which departure times must fit
    within 00:00:00 and 23:59:59 (e.g. 25:45 is 01:45 and 24:35 is 00:35)
    """
    SECONDS_IN_A_DAY = 24 * 3600
    mask = departures.departure_time >= SECONDS_IN_A_DAY
    departures.loc[mask, 'departure_time'] -= SECONDS_IN_A_DAY
    
    
    return departures


def _cut_time_intervals(trip_data, cutoffs,
                        interval_labels, name_of_intervals):
    # Memento: cutoffs are hours and (departure 
    # and arrival) times are parsed as seconds
    bin_edges_in_seconds = 3600 * np.array(cutoffs) 
    trip_data.loc[:, name_of_intervals] = pd.cut(trip_data['departure_time'],
                                                 bins=bin_edges_in_seconds,
                                                 right=False,
                                                 labels=interval_labels,)
    
    trip_data = (trip_data
                 .loc[
                     trip_data[name_of_intervals].notnull()
                     ]
                 .copy()
                )
    
    
    return trip_data


def _get_time_span(s):
    """Takes a str label that represents a time interval and computes
    the corresponding time span in seconds.
    
    If the label looks like HH:MM - HH:MM it uses regular expressions
    to do the maths. 
    
    If the label is a single number like H ou HH, it
    assumes it is an one hour interval.
    
    TO DO: ponder whether the else clause is overly rigid
    """
    # TO DO: is there a cleaner, more elegant expression?
    matches = re.search(r'(\d*\d):(\d\d)\s?-?\s?(\d*\d)?:?(\d\d)?', s)
    if matches.group(3) is not None:
        # MEMENTO: the group method in regular expressions is 1-indexed
        span = (
            int(matches.group(3)) * 60
            + int(matches.group(4))
            - int(matches.group(1)) * 60
            - int(matches.group(2))
        )
    else:
        span = 60 


    return span


def _get_mean_headway(trips_per_interval, col_with_intervals):
    """
    Takes number of trips that initiated within a given time interval 
    and assumes they are spaced evenly.
    
    # TO DO: extract this info directly from feed.stop_times
    """    
    # MEMENTO: col_with_intervals is pandas.Categorical class
    time_spans = (trips_per_interval
                  .astype({col_with_intervals: str})
                  [col_with_intervals]
                  .map(lambda x: _get_time_span(x))
                 )

    headways = time_spans / trips_per_interval.trips
    
    # In time windows with no trips, the above operation returns np.inf,
    # I assume np.nan is better to wrap one's head around and hence
    # the replacement
    mask = (headways==np.inf) | (headways==-np.inf)
    headways = np.where(mask, np.nan, headways)
    
    
    return headways


def _slice_gtfs_by_direction_and_time_window(operational_data, window, direction=None):
    """Takes subset of operational data in order to evaluate 
    level of service by period of day.
    """
    mask = operational_data.window == window
    
    if direction is not None:
        dir_mask = operational_data.direction == direction
        mask = mask & dir_mask
        
    operations_within_time_window = operational_data.loc[mask,:].copy()
    
    
    return operations_within_time_window


def _get_trip_start_data(trip_data):
    """Takes only records related to the time a trip departs its 
    initial stop. Drops remaining records.
    """
    # Stop sequence numbers must increase along the trip but
    # do not need to be consecutive. Also, the first stop in
    # the sequence is not necessarily 1
    first_stop = (trip_data
                  .sort_values('stop_sequence')
                  .pivot_table('stop_sequence',
                               index='trip_id',
                               aggfunc='first')
                  .stop_sequence
                  .unique()
                 )
                  
    try:
        # first_stop should be a one element array because,
        # usually, the id number for the first stop should be
        # the same throughout. If not, int(first_stop) will raise
        # an exception because it cannot handle arrays with more than one
        # element. Such a case most likely demands further investigation.
        first_stop = int(first_stop)
        first_departures = trip_data.loc[trip_data.stop_sequence == first_stop]
    except TypeError:
        print(f"There is an inconsistency in stop sequence ids")
        
        
    return first_departures


def _hourly_trip_summary(trip_data, agg_on):
    HOURS_IN_DAY = range(25)
    one_hour_labels = [str(h) + ':00' for h in HOURS_IN_DAY[:-1]]
    
    hourly_trips = (departures
                    .pipe(_cut_time_intervals,
                          HOURS_IN_DAY,
                          one_hour_labels,
                          'hour')
                    .pipe(_agg_trips,
                          agg_on,
                          'hour')
                   )
    
    new_name = {'trips': 'max_hourly_trips'}
    max_hourly_trips = (hourly_trips
                        .rename(columns=new_name)
                        .pivot_table('max_hourly_trips',
                                     index=[agg_on, 'direction_id'],
                                     aggfunc='max',)
                        .reset_index()
                       )
    
    hourly_trips['min_hourly_headway'] = _get_mean_headway(hourly_trips,
                                                           'hour',)
    min_hourly_headway = (hourly_trips
                          .pivot_table('min_hourly_headway',
                                       index=[agg_on, 'direction_id'],
                                       aggfunc='min',)
                          .reset_index()
                         )
    
    
    return max_hourly_trips, min_hourly_headway


def _time_window_summary(trip_data, cutoffs, agg_on):    
    trips_per_window = (trip_data
                        .pipe(_cut_time_intervals,
                              cutoffs,
                              _get_window_labels(cutoffs),
                              'window')
                        .pipe(_agg_trips,
                              agg_on,
                              'window')
                       )
    
    trips_per_window['headway_minutes'] = _get_mean_headway(trips_per_window,
                                                            'window')
    
    
    return trips_per_window


# TEXT HANDLINGS AND PARSINGS -------------------------------------------------
# -----------------------------------------------------------------------------
def _get_window_labels(cutoffs):
    
    # TO DO: allow for float inputs (e.g. 5.5 as in 05:30:00)
    def _create_label(i):
        cut = cutoffs[i]
        try:
            hours = int(cut // 1)
            minutes = int(cut % 1 * 60)
            
            label = (datetime
                     .time(hours, minutes)
                     .strftime("%H:%M"))
        except:
            label = '24:00'
        return label
    
    number_of_cuts = len(cutoffs)    
    labels = [
        _create_label(i) + ' - ' + _create_label(i+1)
        for i
        in range(number_of_cuts-1)
             ]
    
    
    return labels


def _get_route_full_names(trip_data):
    names = (trip_data
             .groupby('route_id')
             [['route_short_name', 'route_long_name']]
             .first()
            )
    
    names.loc[names.route_short_name.isnull()] = '---'
    names.loc[names.route_long_name.isnull()] = '---'  
    names['route_name'] = [short + ' ' + long \
                           for short,long \
                           in zip(names.route_short_name, names.route_long_name)]
    
    
    return names[['route_name']]


def summarize_trips(feed, summ_by, cutoffs):
    """Takes the stop_times DataFrame, as returned by parse_gtfs()
    and returns a summary of trips by the time windows of choice.
    It allows comparison with the most loaded hour within the day.
    
    Parameters
    ----------
    stop_times : DataFrame
    summ_by : str {'route_id', 'stop_id'}
    cutoffs : list
        List of integers representing time interval limits
        
    Returns
    -------
    summary : DataFrame
    """
    departures = (
        _assemble_operational_data(feed)
        .pipe(_fix_departure_times)
        .pipe(_get_trip_start_data)
    )
    
    with pd.option_context('mode.chained_assignment', None):
        hourly_summary = _hourly_trip_summary(departures,
                                              agg_on=summ_by,)
        max_hourly_trips, min_hourly_headway = hourly_summary

        summary = (departures
                   .pipe(_time_window_summary, cutoffs, summ_by)
                   .merge(max_hourly_trips, how='left')
                   .merge(min_hourly_headway, how='left')
                  )
    
    if summ_by == 'route_id':
        summary = summary.merge(_get_route_full_names(departures),
                                how='left',
                                left_on=summ_by,
                                right_index=True,)
    elif summ_by == 'stop_id':
        summary = summary.merge(departures[['stop_id', 'stop_name']],
                                how='left',)
        
    # Final polishments and embelishments
    preferred_column_order = [summ_by, 'direction', 'route_name',
                              'window', 'trips', 'headway_minutes', 
                              'max_hourly_trips', 'min_hourly_headway',]
    
    summary = (summary
               .rename(columns={'direction_id': 'direction'})
               .replace({'direction': {0: 'Inbound', 1: 'Outbound'}})
               .reindex(columns=preferred_column_order)
               .sort_values([summ_by, 'window'])
              )
    # TO DO: 0 for inbound and 1 for outbound is too
    # specific for the BH case! Unhardcode it!
        
    
    return summary