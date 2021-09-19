import datetime
import warnings

import geopandas as gpd
import pandas as pd
import partridge as ptg


   
def _put_data_into_stop_times(stop_times, trips, routes, stops):
    """This is just to facilitate some data manipulations.
    I did thisbecause this project is largely inspired
    by https://github.com/Bondify/gtfs_functions and I
    followed along, but this may change in the future.
    """
    trips = trips.merge(routes, how='left')
    stop_times = stop_times.merge(trips, how='left')
    stop_times = stop_times.merge(stops, how='left')


    return stop_times


def _get_window_labels(cutoffs):
    
    # TO DO: allow for float inputs (e.g. 5.5 as in 05:30:00)
    def _create_label(i):
        cut = cutoffs[i]
        try:
            # Accepts only integers from 0 to 23
            label = datetime.time(cut).strftime("%H:%M") 
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
    names = trip_data.groupby('route_id')[['route_short_name', 'route_long_name']].first()
    
    names.loc[names.route_short_name.isnull()] = '---'
    names.loc[names.route_long_name.isnull()] = '---'  
    names['route_name'] = [short + ' ' + long \
                           for short,long \
                           in zip(names.route_short_name, names.route_long_name)]
    
    
    return names[['route_name']]


def _get_shapes_into_route_summary(stop_times, route_summary, route_shapes):
    """
    Makes the correspondence between routes and their shapes
    in a way that specifically handles the structure of the DataFrame
    returned by .gtfs.summarize_trips()
    """
    id_cols = ['route_id', 'direction_id', 'shape_id']
    shape_correspondence = stop_times.drop_duplicates(subset=id_cols)
    
    shape_correspondence = shape_correspondence.reindex(columns=id_cols)
    route_geospatial_summary = route_summary.merge(shape_correspondence,
                                                   how='left',)
    route_geospatial_summary = route_shapes.merge(route_geospatial_summary,
                                                  how='right',)
    
    route_geospatial_summary.to_crs(epsg=4326, inplace=True)
    
    
    return route_geospatial_summary


def _get_route_stops(stop_times):
    """Finds the transit (or bus) stops that each route passes by
    in the order established by the operational schedule.
    
    Parameters
    ----------
    stop_times: DataFrame
        Stop times (as returned from assemble_gtfs_files())
        
    Returns
    -------
    stop_sequence: DataFrame
    """
    stop_sequence = stop_times.drop_duplicates(subset=['stop_id', 'stop_name', 
                                                       'stop_sequence', 'shape_id',
                                                      ])
    stop_sequence = stop_sequence.reindex(columns=['route_id', 'direction_id',
                                                   'shape_id', 'stop_id',
                                                   'stop_name', 'stop_sequence',
                                                   'geometry',
                                                  ])
    stop_sequence = gpd.GeoDataFrame(stop_sequence,
                                     crs='EPSG:4326',
                                     geometry='geometry')
    # The following sorting is just for sanity's sake
    stop_sequence.sort_values(['route_id', 'direction_id', 
                               'shape_id', 'stop_sequence'],
                              inplace=True,
                             ) 
    stop_sequence.to_crs(epsg=31983, inplace=True)
    
    
    return stop_sequence


def _get_route_shapes(stop_sequence, route_shapes):
    """Merges output from _get_route_stops() with the 
    corresponding route shapes.
    """
    shapes = route_shapes.to_crs(epsg=31983)
    shapes.rename(columns={'geometry': 'route_geometry'}, inplace=True)
    shapes.set_geometry('route_geometry', crs='EPSG:31983', inplace=True)
    
    stop_sequence = stop_sequence.merge(shapes, how='left')
    
    
    return stop_sequence


def _assemble_route_segments_dataframe(route_segments, stop_ids, stop_names,
                                       route_id, direction_id, shape_id):
    segments_and_data = {
        'start_stop_id': stop_ids[:-1],
        'end_stop_id': stop_ids[1:],
        'start_stop_name': stop_names[:-1],
        'end_stop_name': stop_names[1:],
                      }
    segments_and_data = pd.DataFrame.from_dict(segments_and_data)

    segments_and_data = route_segments.merge(segments_and_data,
                                             how='left',
                                             left_index=True,
                                             right_on='end_stop_id',
                                            )
    
    
    segments_and_data['route_id'] = route_id
    segments_and_data['direction_id'] = direction_id
    segments_and_data['shape_id'] = shape_id
    # TO DO: Is a stop sequence column necessary?
    
    segments_and_data['route_length_m'] = segments_and_data.geometry.map(lambda x: x.length)
    segments_and_data['segment_id'] = segments_and_data[['start_stop_id', 'end_stop_id']].apply(
        lambda x: x[0] + '-' + x[1],
        axis=1,
                                                                                               )
    
    segments_and_data = segments_and_data.reindex(columns=['route_id', 'direction_id',
                                                           'shape_id','segment_id',
                                                           'start_stop_name', 'end_stop_name',
                                                           'start_stop_id','end_stop_id',
                                                           'route_length_m', 'segments',
                                                          ]
                                                 )
    segments_and_data.set_index(['route_id', 'direction_id', 'shape_id'], inplace=True)   

    segments_and_data = gpd.GeoDataFrame(segments_and_data, geometry='segments', crs="EPSG:31983")
    segments_and_data.to_crs(epsg=4326, inplace=True)
    
    
    return segments_and_data