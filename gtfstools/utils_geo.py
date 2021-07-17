import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Point
from scipy.stats import zscore



def _get_translation_distances(vertices, stop_to_project, threshold):
    translation_distances = {}
    for i,vertex in enumerate(vertices):
        try:
            growing_route = LineString(vertices[ : i+2])
        except ValueError:
            # When vertices idx is out of range by the time the 
            #loop is ending
            break
        
        distance = growing_route.distance(stop_to_project)
        if distance <= threshold:
            translation_distances[i] = growing_route.distance(stop_to_project)

    
    return translation_distances


def _assert_projection_consistency(translation_distances, stop_locations,
                                    k, vertices, route_id):
    """Checks if the smallest translation distance corresponds to a consistent
    projection, which is one that lies between the projections of the previous 
    and of the subsequent transit (or bus) stop.
    
    Parameters
    ----------
    translation_distances : dict
        Dictionary whose keys correspond to the stop sequence on the route, and
        whose values are the corresponding translation distance
    stop_locations : array-like
        Iterable containing shapely Point objects representing transit stop locations
    k : int
        Key of the stop being checked for consistency
    vertices : array-like
        Iterable containing point coordinates of the route
    route_id : str
        Route identifier
        
    Returns
    -------
    idx : int
        Used as key in translation distances -> smallest (valid) distance.
        
        Used (appropriately) with vertices -> gets route segment where
        (valid) projection is made.
    anom_type : int {0, 1, 2}
        Type of anomalous behaviour detected
        0 -> checked and consistent
        1 -> stop too far away from route
        2 -> projections are (most likely) inconsistent
    """
    def _project_stop_and_surroundings():
        # TO DO: extract ("un-nest") this function?
        # Raises exception if dict keys are popped 'till it's empty,
        #which indicates projection errors
        idx = min(distances, key=distances.get)
        line = LineString(vertices[ : idx+2])
        kth_stop_proj = line.project(stop_locations[k])
        
        try:
            next_stop_proj = line.project(stop_locations[k+1])
        except IndexError:
            next_stop_proj = np.nan
            
        try:
            previous_stop_proj = line.project(stop_locations[k-1])
        except IndexError:
            previous_stop_proj = np.nan
        
        return idx, previous_stop_proj, kth_stop_proj, next_stop_proj
    
    number_of_segments = len(vertices) - 1
    
    distances = translation_distances.copy()
    
    projections = _project_stop_and_surroundings()
    idx, previous_stop_proj, kth_stop_proj, next_stop_proj = projections
    
    if np.isnan(previous_stop_proj) & np.isnan(next_stop_proj):
        raise Exception(
            "There's only one (apparently) valid transit stop in "\
            f"route {route_id}, which doesn't really make sense."\
                       )
        
    else:
        anom_type = 0
        distances.pop(idx)
        # TO DO: should this check be more strict? Maybe use AND, instead of OR
        while (kth_stop_proj > next_stop_proj) | (kth_stop_proj < previous_stop_proj):    
            if distances:
                projections = _project_stop_and_surroundings()
                idx, previous_stop_proj, kth_stop_proj, next_stop_proj = projections
                
                distances.pop(idx)
            
            else:
                idx = min(translation_distances, key=translation_distances.get)
                anom_type = 2
                
                break
    
    return idx, anom_type
    


def _project_stop_on_route(route_shape, stop_locations, stop_ids, route_id):
    """There are instances in which a segment at the start of a route is too close
    to a segment at the end, that might lead to bus stops being wrongly projected
    onto the wrong segment. This function addresses this issue.
    
    TO DO: I erred on the side of caution, which means that this code probably
    be made more efficient after due consideration
    
    Parameters
    ----------
    
    """
    vertices = list(route_shape.coords)
    
    number_of_stops = len(stop_locations)
    stop_keys = range(number_of_stops)
    stop_keys_and_ids = {k: idx for k,idx in zip(stop_keys, stop_ids)}

    output_holder = {}
    for k,stop_id in stop_keys_and_ids.items():
        stop_to_project = stop_locations[k]
        
        threshold = 100 # Arbitrarily chosen
        translation_distances = _get_translation_distances(vertices,
                                                           stop_to_project,
                                                           threshold,
                                                          )
        
        if not translation_distances:
            # TO DO: this might be too verbose
            #print(
            #    f"Stop {k} for route {route_id} is always more than "\
            #    f"{threshold} m away from route {route_id}" \
            #    )
            
            output_holder[k] = {
                'stop_id':stop_id,
                'projected_point': np.nan,
                'translation_distance': np.nan,
                'anomaly_type': 1,
                               }
    
        else:
            idx, anom_type = _assert_projection_consistency(translation_distances, 
                                                            stop_locations, 
                                                            k,
                                                            vertices,
                                                            route_id,
                                                            )
            route_segment = LineString(vertices[idx : idx+2])
            stop_projection = route_segment.project(stop_to_project)
            output_holder[k] = {
                'stop_id': stop_id,
                'projected_point': route_segment.interpolate(stop_projection),
                'translation_distance': translation_distances[idx],
                'anomaly_type': anom_type,
                               }
        
    output_holder = pd.DataFrame.from_dict(output_holder, orient='index')
    output_holder.index.name = 'stop'
    
    
    return output_holder


def _complex_split(line, stop):
    """
    taken from: https://gis.stackexchange.com/a/214432
    """
    coords = list(line.coords)  
    distance_to_stop = line.project(stop)

    # Projected distances are never negative or greater than line length,
    #but the inequalities ~ might ~ account for floating point imprecisions
    if distance_to_stop <= 0:
        return [Point(coords[0]), line]
    elif distance_to_stop >= line.length:
        return [line, line]
    for i,vertex in enumerate(coords):
        vertex = Point(vertex)
        distance_to_vertex = line.project(vertex)
        if distance_to_vertex==distance_to_stop:
            return [LineString(coords[:i+1]), LineString(coords[i:])]
        elif distance_to_vertex>distance_to_stop:
            break_point = line.interpolate(distance_to_stop)
            break_point = list(break_point.coords)
            return [
                LineString(coords[:i] + break_point),
                LineString(break_point + coords[i:])
                   ]

        
def _cut_route_into_segments(route, projections):
    line = LineString(route) # makes a copy and ensures orginal shape remains intact
    
    projected_stops = projections['projected_point']
    stop_ids = projections['stop_id']
    
    segments = {}
    for idx,stop in zip(stop_ids, projected_stops):
        try:
            splitted_line = _complex_split(line, stop)
            segments[idx] = splitted_line[0]
            line = splitted_line[1]
        except AttributeError:
            # For cases in which stop = nan — refer to _assert_projection_consistency()
            #to better grasp this issue
            
            # TO DO: get straight line distance between unprojected stops
            pass
    
    segments = pd.DataFrame.from_dict(segments, orient='index', columns=['segments'])
    segments = gpd.GeoDataFrame(segments, geometry='segments', crs='EPSG:31983')
    segments.drop(segments.index[0], inplace=True)
    
    
    return segments


def _flag_local_outliers(distances, stop_sequence, route_id, direction_id, 
                         shape_id, stop_ids, threshold):
    """Due to errors during measurement or entry phases, stops do
    not perfectly align with the route. This calculates the distance from 
    each stop to its corresponding route and flags outliers.
    
    After a z-score standardization, outliers are assumed to be those that 
    are 2.5 standard deviations (or more) from the mean.
    """

        
    distances_holder = {'stop_id': stop_ids,
                        'distance_m': distances,
                        'local_z_score': zscore(distances, nan_policy='omit'),
                        }
    distances_df = pd.DataFrame.from_dict(distances_holder)

    distances_df['local_outlier'] = distances_df.distance_m.map(
        lambda x:
        True if x>=threshold or x<=(-threshold)
        else False
                                                                )
    
    distances_df['route_id'] = route_id
    distances_df['direction_id'] = direction_id
    distances_df['shape_id'] = shape_id
    
    distances_df.set_index(['route_id', 'direction_id', 'shape_id'], inplace=True)
    distances_df.drop(columns='local_z_score', inplace=True)
    
    
    return distances_df


