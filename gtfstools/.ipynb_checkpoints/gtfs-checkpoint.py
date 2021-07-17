import pandas as pd

from scipy.stats import zscore

from . import utils_parsing as up
from . import utils_time as ut
from . import utils_geo as ug


def parse_feed(path):
    """
    Reads gtfs files and, after some manipulations, puts all 
    relevant info into stop_times data.
    """
    gtfs_feed, routes, trips, stops, stop_times, shapes = up._read_gtfs_feed(path)
    stop_times = up._put_data_into_stop_times(stop_times, trips, routes, stops)
    
    
    return gtfs_feed, routes, trips, stops, stop_times, shapes


def summarize_trips(stop_times, summ_by, cutoffs):
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
    departures = stop_times.copy()
    departures = ut._fix_departure_times(departures)
    departures = ut._get_trip_start_data(departures)
    
    max_hourly_trips, min_hourly_headway = ut._hourly_trip_summary(departures,
                                                                   agg_on=summ_by,)
    
    trips_per_window = ut._time_window_summary(departures,
                                               cutoffs,
                                               agg_on=summ_by,)    
    
    summary = trips_per_window.merge(max_hourly_trips,
                                     how='left',)    
    summary = summary.merge(min_hourly_headway,
                            how='left',)
    
    if summ_by == 'route_id':
        route_names = up._get_route_full_names(departures)
        summary = summary.merge(route_names,
                                how='left',
                                left_on=summ_by,
                                right_index=True,)
    # TO DO: elif for getting stop names
        
    # Final polishments and embelishments
    summary.rename(columns={'direction_id': 'direction'},
                         inplace=True,)
    
    summary.replace({'direction': {0: 'Inbound', 1: 'Outbound'}},
                          inplace=True,)
    
    preferred_column_order = [summ_by, 'direction', 'route_name',
                              'window', 'trips', 'headway_minutes', 
                              'max_hourly_trips', 'min_hourly_headway',]
    summary = summary.reindex(columns=preferred_column_order)
    
    summary.sort_values([summ_by, 'window'], inplace=True)
        
    
    return summary


def cut_routes(stop_times, route_shapes, flag_outliers=True, threshold=2.5):
    stop_sequence = up._get_route_stops(stop_times)
    stop_sequence = up._get_route_shapes(stop_sequence, route_shapes)

    collection_of_all_segments = []
    collection_of_all_anomalies = []
    for idx,group in stop_sequence.groupby(['route_id', 'direction_id', 'shape_id']):
        route_id, direction_id, shape_id = idx

        route = group.iloc[0,-1]    
        stop_locations = group.geometry.to_numpy()
        stop_names = group.stop_name.to_numpy()
        stop_ids = group.stop_id.to_numpy()
        
        # TO DO: input stop_ids so as to ensure index correspondence
        projections = ug._project_stop_on_route(route, stop_locations,
                                             stop_ids, route_id,
                                            )
        
        route_segments = ug._cut_route_into_segments(route, projections)

        segments_and_data = up._assemble_route_segments_dataframe(route_segments, stop_ids, 
                                                                 stop_names, route_id,
                                                                 direction_id, shape_id,
                                                                 )
        collection_of_all_segments.append(segments_and_data)

        if flag_outliers:
            distances = projections['translation_distance'].to_numpy()
            anomalous_projections =  ug._flag_local_outliers(distances, stop_sequence, route_id,
                                                             direction_id, shape_id,
                                                             stop_ids, threshold)
            # TO DO: check consistency of following line... seems ok, though
            anomalous_projections['anom_type'] = projections.anomaly_type.to_numpy()
            
            collection_of_all_anomalies.append(anomalous_projections)

    cut_routes = pd.concat(collection_of_all_segments)

    if flag_outliers:
        anomalies = pd.concat(collection_of_all_anomalies)
        anomalies['global_z_score'] = zscore(anomalies.distance_m, nan_policy='omit')
        anomalies['global_outlier'] = anomalies.distance_m.map(
            lambda x:
            True if x>=threshold or x<=(-threshold)
            else False
                                                  )
        anomalies.drop(columns='global_z_score', inplace=True)
        return cut_routes, anomalies
    else:
        return cut_routes


