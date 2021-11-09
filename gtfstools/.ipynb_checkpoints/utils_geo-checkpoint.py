"""Geospatial utility functions"""


import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import linemerge
from scipy.stats import zscore

from .readers import _read_shapes
from . import summaries as summ


def _cut(line, point_to_project):
    distance = line.project(point_to_project)
    
    if distance <= 0.0:
        return [LineString(line), 'u']
    
    if distance >= line.length:
        return [LineString(line), 'v']
    
    coords = list(line.coords)
    for i, coord in enumerate(coords):
        proj_dist = line.project(Point(coord))
        
        if proj_dist == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        
        if proj_dist > distance:
            projection = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(projection.x, projection.y)]),
                LineString([(projection.x, projection.y)] + coords[i:])]
        
        
def get_road_network(feed, epsg):
    west, south, east, north  = feed.stops.total_bounds
    roads = ox.graph_from_bbox(north,
                               south,
                               east,
                               west,
                               network_type='drive',)
    
    roads = ox.project_graph(roads, to_crs=epsg)
    
    nodes, edges = ox.graph_to_gdfs(roads)
    # This may look redundant, but it is meant to fill in all edge geometry
    # attributes (see https://stackoverflow.com/a/64376567/15994934).
    # On first thought, it seemed more efficient than working with dataframes.
    roads = ox.graph_from_gdfs(nodes, edges, graph_attrs=roads.graph)
    
    
    return roads
    


def _proj_stops_on_roads(feed, roads):
    """This projects bus stops on the nearest edge of the road
    network, while ensuring the correspondence between node ids
    (in the network) and stop ids (in the GTFS feed).
    """
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        stops = feed.stops
    
    roads, stop_aliases = _fix_ids(roads, stops)
    
    nearest_edges = ox.nearest_edges(roads,
                                     X=stops.geometry.x.to_list(),
                                     Y=stops.geometry.y.to_list(),)
    
    last_edge_idx = edges.osmid.max()
    for i, edge_idx in enumerate(nearest_edges):
        u, v, key = edge_idx
        
        stop = stops.geometry.iloc[0]
        stop_id = stops.stop_id.iloc[0]
        stop_alias = stop_aliases[stop_id]
        
        edge_geometry = roads[u][v][key]['geometry']
        segments = _cut(edge_geometry, stop)
        
        if segments[1] == 'u':
            nx.relabel_nodes(index={u: stop_alias})
        elif segments[1] == 'v':
            nx.relabel_nodes(index={v: stop_alias})
        else:
            base_attr = roads[u][v][key].copy()
            
            base_attr['geometry'] = segments[0]
            base_attr['length'] = segments[0].length
            nx.add_edges_from(
                              roads,
                              [(u, stop_alias, 0, base_attr)]
                             )
            
            last_edge_idx += 1
            geometry['osmid'] = last_edge_idx
            base_attr['geometry'] = segments[1]
            base_attr['length'] = segments[1].length
            nx.add_edges_from(
                              roads,
                              [(stop_alias, v, 0, base_attr)]
                             )
            
            roads.remove_edge(u, v)
            
    attr = {node: {'is_transit_stop': 1}
            for node
            in stop_aliases.values()}
    nx.set_node_attributes(roads, attr)
            
            
    return roads, stop_aliases


def _get_shortest_path_lines(stop_sequence, roads, epsg):
    """For each route, it draws a LineString that contains all stops
    therein. The line is drawn in such a way that it minimizes inter-
    station distance.
    """    
    shapes = {} 
    for route_id, group in stop_sequence.groupby(['route_id', 'direction_id']):
        orig_stops = group.alias_id.iloc[:-1].to_list()
        dest_stops = group.alias_id.iloc[1:].to_list()
        
        shortest_path = ox.distance.shortest_path(roads,
                                                  orig=orig_stops,
                                                  dest=dest_stops,
                                                  weight='length',)
        
        line_segments = []
        for node_sequence in shortest_path:
            coords = [(roads.nodes[n]['x'], roads.nodes[n]['y'])
                      for i
                      in n in node_sequence]
            
            line_segments.append(LineString(coords))
        
        line = MultiLineString(line_segments)
        line = linemerge(line)
        
        shapes[route_id].append(line)
            
    shapes = pd.DataFrame.from_dict(shapes,
                                    orient='index',
                                    columns=['geometry'])
    
    shapes = gpd.GeoDataFrame(shapes,
                              crs=f'EPSG:{epsg}',
                              geometry='geometry')
    shapes.index.name = 'shape_id'
    
    
    return shapes.reset_index()


def build_routes(feed, road_network=None, epsg=5641):
    """If the base GTFS dataset does not contain shapes.txt,
    build the shapes for each route assuming that the line
    segment between each pair of stops is the shortest road
    network path between them. This latter is not ideal but
    it is a necessary, and even reasonable approximation
    in some circunstances.
    """
    stop_sequence = _stop_locations_by_route(feed, epsg)
    stop_sequence.to_crs(epsg=epsg, inplace=True)
    
    if road_network is None:
        road_network = get_road_network(feed, epsg)
        
    road_network, stop_aliases = _proj_stops_on_roads(feed, road_network)
    
    stop_aliases = pd.DataFrame.from_dict(stop_aliases,
                                          orient='index',
                                          columns=['alias_id'])
    stop_sequence = stop_sequence.merge(stop_aliases,
                                        how='left',
                                        left_on='stop_id',
                                        right_inex=True)
    shapes = _get_shortest_path_lines(stop_sequence, road_network, epsg)
    
    feed.trips['shape_id'] = shapes.trips.route_id
    
    
    return feed


#def _get_route_shapes(stop_sequence, route_shapes):
#    """Merges output from _get_route_stops() with the 
#    corresponding route shapes.
#    """
#    shapes = route_shapes.to_crs(epsg=31983)
#    shapes.rename(columns={'geometry': 'route_geometry'}, inplace=True)
#    shapes.set_geometry('route_geometry', crs='EPSG:31983', inplace=True)
#    
#    stop_sequence = stop_sequence.merge(shapes, how='left')
#    
#    
#    return stop_sequence


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
        # which indicates projection errors
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

