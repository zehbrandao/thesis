import re

import numpy as np
import pandas as pd
import geopandas as gpd

from .utils_parsing import _get_window_labels


# GENERAL PURPOSE ------------------------------------------------------------
def _fix_departure_times(departures):
    """Some departures correspond to trips that initiated the day
    before, hence they are recorded as, e.g., 25:45 or 24:35, which
    might have its uses.
    
    This function is aimed at cases in which departure times must fit
    within 00:00:00 and 23:59:59 (e.g. 25:45 is 01:45 and 24:35 is 00:35)
    """
    SECONDS_IN_A_DAY = 24*3600
    mask = departures.departure_time>=SECONDS_IN_A_DAY
    departures.loc[mask, 'departure_time'] -= SECONDS_IN_A_DAY
    
    
    return departures


def _cut_time_intervals(trip_data, cutoffs,
                        interval_labels, name_of_intervals):
    # TO DO: Allow for cutoffs that are not full hours
    #
    # The following line if because cutoffs are hours and we 
    # need to cut a column with seconds.
    bin_edges_in_seconds = 3600*np.array(cutoffs) 
    trip_data.loc[:,name_of_intervals] = pd.cut(trip_data['departure_time'], 
                                                bins=bin_edges_in_seconds, 
                                                right=False,
                                                labels=interval_labels,)
    
    trip_data = trip_data.loc[trip_data[name_of_intervals].notnull()].copy()
    
    
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
    # TO DO: consider a cleaner, more elegant expression
    matches = re.search(r'(\d*\d):(\d\d)\s?-?\s?(\d*\d)?:?(\d\d)?', s)
    if matches.group(3) is not None:
        # MEMENTO: the group method in regular expressions is 1-indexed
        span = int(matches.group(3))*60   \
               + int(matches.group(4))    \
               - int(matches.group(1))*60 \
               - int(matches.group(2))

    else:
        span = 60 


    return span


def _get_average_headway(trips_per_interval, col_with_intervals):
    """
    Takes number of trips that initiated within a given time interval 
    and assumes they are spaced evenly.
    
    # TO DO: extract this info directly from feed.stop_times
    """    
    # MEMENTO: col_with_intervals is pandas.Categorical class
    trips_per_interval = trips_per_interval.astype({col_with_intervals: str})
    time_spans = trips_per_interval[col_with_intervals].map(lambda x: _get_time_span(x))
    headways = time_spans / trips_per_interval.trips
    
    # In time windows with no trips, the above operation returns np.inf,
    # I assume np.nan is better to wrap one's head around and hence make
    # the replacement
    mask = (headways==np.inf) | (headways==-np.inf)
    headways = np.where(mask, np.nan, headways)
    
    
    return headways


def _slice_gtfs_by_direction_and_time_window(operational_data, window, direction=None):
    """Takes subset of operational data in order to evaluate 
    level of service by period of day.
    """
    mask = operational_data.window==window
    
    if direction is not None:
        dir_mask = operational_data.direction==direction
        mask = mask & dir_mask
        
    operations_within_time_window = operational_data.loc[mask,:].copy()
    
    
    return operations_within_time_window


# TRIP WRANGLING -------------------------------------------------------------
def _get_trip_start_data(trip_data):
    """Takes only records related to the time a trip departs its 
    initial stop. Drops remaining records.
    """
    # The code I'm basing myself on assumes the first stop is labeled 1, and it does not seem to be the case, necessarily
    first_stop = trip_data.pivot_table('stop_sequence',
                                        index='trip_id',
                                        aggfunc='first')
    first_stop = first_stop.stop_sequence.unique()
    
    if len(first_stop)==1:
        first_stop = int(first_stop)
        first_departures = trip_data.loc[trip_data.stop_sequence==first_stop]
    else:
        raise Exception(f"There's an inconsistency in stop sequence ids")
        
        
    return first_departures


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
    trip_aggregation = trip_data.pivot_table(
        
        'trip_id',
        index=[agg_on, 'direction_id', col_with_intervals],
        aggfunc='count',
                                             
                                            )
    
    trip_aggregation.reset_index(inplace=True)
    trip_aggregation.rename(columns={'trip_id': 'trips'}, inplace=True)
    
    
    return trip_aggregation


def _hourly_trip_summary(trip_data, agg_on):
    HOURS_IN_DAY = range(25)
    one_hour_labels = [str(h) + ':00' for h in HOURS_IN_DAY[:-1]]
    departures = _cut_time_intervals(trip_data,
                                     HOURS_IN_DAY,
                                     one_hour_labels,
                                     'hour',)
    hourly_trips = _agg_trips(trip_data=departures,
                              agg_on=agg_on,
                              col_with_intervals='hour',)
    
    new_name = {'trips': 'max_hourly_trips'}
    max_hourly_trips = hourly_trips.rename(columns=new_name)
    max_hourly_trips = max_hourly_trips.pivot_table('max_hourly_trips',
                                                    index=[agg_on, 'direction_id'],
                                                    aggfunc='max',)
    max_hourly_trips.reset_index(inplace=True)
    
    hourly_trips['min_hourly_headway'] = _get_average_headway(hourly_trips,
                                                              'hour',)
    min_hourly_headway = hourly_trips.pivot_table('min_hourly_headway',
                                                   index=[agg_on, 'direction_id'],
                                                   aggfunc='min',)
    min_hourly_headway.reset_index(inplace=True)
    
    
    return max_hourly_trips, min_hourly_headway


def _time_window_summary(trip_data, cutoffs, agg_on):
    window_labels = _get_window_labels(cutoffs)
    trips_per_window = _cut_time_intervals(trip_data, cutoffs, window_labels, 'window')    
    trips_per_window = _agg_trips(trip_data=trips_per_window,
                                  agg_on=agg_on,
                                  col_with_intervals='window',)    
    trips_per_window['headway_minutes'] = _get_average_headway(trips_per_window, 'window')
    
    
    return trips_per_window



