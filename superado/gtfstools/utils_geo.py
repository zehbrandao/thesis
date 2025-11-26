"""Geospatial utility functions"""
# TO DO: find a way to automatically get the best projected CRS
# for the GTFS location


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


# OPERATIONS WITH THE ROAD NETWORK --------------------------------------------
# -----------------------------------------------------------------------------
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
    stop_sequence = summ._stop_locations_by_route(feed, epsg)
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


# OPERATIONS WITH THE ROUTE LINESTRING ----------------------------------------
# -----------------------------------------------------------------------------
def _project_stop_on_route(group, threshold=100):
    line = group.iloc[0].geometry
    
    proj_collection = {}
    for k, (stop_id, loc) in enumerate(zip(group.stop_id, group.geometry)):
        translation_distances = _get_translation_distances(line,
                                                           stop_id,
                                                           loc,
                                                           threshold)
        
        if not translation_distances:
            proj_collection[stop_id] = {'projected_point': np.nan,
                                        'translation_distance': np.nan,
                                        'anomaly_type': 1}
        else:








def _get_translation_distances(line, stop_loc, threshold):
    vertices = list(line.coords)
        
    trans_dist = {}
    for i in range(2, len(vertices) + 1):
        growing_route = LineString(vertices[:i])
        distance = growing_route.distance(stop_loc)
        
        if distance <= threshold:
            trans_dist[i] = growing_route.distance(stop_loc)

    
    return trans_dist


def _find_true_projection(translation_distances, group,
                          k, line, route_id):
    """The correct projection of the stop on the route does not
    necessarily correspond to the smallest translation distance,
    given innacuracies in stop geolocation. This implemnts one
    way to handle this issue.
    
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
        """
        """
        i = min(trans_dist, key=trans_dist.get)
        line_stretch = LineString(line.coords[:i])
        
        k_proj = line_stretch.project(group.iloc[k].geometry)
        
        try:
            next_stop_proj = line.project(group.iloc[k + 1].geometry)
        except IndexError:
            next_stop_proj = np.nan
            
        try:
            previous_stop_proj = line.project(stop_locations[k-1])
        except IndexError:
            previous_stop_proj = np.nan
        
        return i, previous_stop_proj, k_proj, next_stop_proj
    
    
    trans_dist = translation_distances.copy()
    
    projections = _project_stop_and_surroundings()
    i, previous_stop_proj, kth_stop_proj, next_stop_proj = projections
    
    if np.isnan(previous_stop_proj) & np.isnan(next_stop_proj):
        route_id = group.iloc[k].route_id
        
        raise Exception("There's only one (apparently) valid transit stop in "
                        f"route {route_id}, which doesn't really make sense.")
    
    else:
        anom_type = 0
        trans_dist.pop(i)
        # TO DO: should this check be more strict? Maybe use AND, instead of OR
        while (kth_stop_proj > next_stop_proj) | (kth_stop_proj < previous_stop_proj):    
            if trans_dist:
                projections = _project_stop_and_surroundings()
                idx, previous_stop_proj, kth_stop_proj, next_stop_proj = projections
                
                trans_dist.pop(idx)
            
            else:
                idx = min(trans_dist, key=trans_dist.get)
                anom_type = 2
                
                break
    
    return idx, anom_type
    


def _project_stop_on_route(group, threshold=100):
    """Projects bus stop into route LineString if it is within the
    imposed threshold. It takes into account the following issue:
    
        There are routes where stretches at the beginning are very
        close to stretches in the end. On top of that, innacuracies
        in stop geolocation may cause for a stop at the beggining
        to be closer to the stretch by the end of the line. 
        
        This function corrects this issue. I chose to err on the side
        of caution, which means that there may be a more efficient way.
    
    Parameters
    ----------
    """
    #route_id, direction_id, shape_id = idx
    
    # Each row of group contains the route LineString. It does that
    # because the library that inspired this one did so too.
    #
    # TO DO: avoid the above repetition
    line = group.iloc[0].geometry
    
    proj_collection = {}
    for k, (stop_id, loc) in enumerate(zip(group.stop_id, group.geometry)):
        translation_distances = _get_translation_distances(line,
                                                           stop_id,
                                                           loc,
                                                           threshold)
        
        if not translation_distances:
            proj_collection[stop_id] = {'projected_point': np.nan,
                                        'translation_distance': np.nan,
                                        'anomaly_type': 1}
        else:
            
        
    
    
    
    
    
    
    
    
    
    stop_locations = group.geometry.to_numpy()
    number_of_stops = len(stop_locations)
    stop_ids = group.stop_id.to_numpy()
    stop_aliases = {k: idx
                    for k, idx
                    in zip(range(number_of_stops), stop_ids)}

    route_shape = group.iloc[0].geometry    
    vertices = list(route_shape.coords)

    output_holder = {}
    for k, stop_id in stop_aliases.items():
        stop_to_project = stop_locations[k]
        
        threshold = 100 # Arbitrarily chosen
        translation_distances = _get_translation_distances(group.iloc[0].geometry,
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
            idx, anom_type = _find_true_projection(translation_distances, 
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
            # For cases in which stop = nan — refer to _find_true_projection()
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



def cut_routes(feed, flag_outliers=True, threshold=2.5, epsg=5641):
    stop_sequence = summ._stop_locations_by_route(feed, epsg)
    stop_sequence.to_crs(epsg=epsg, inplace=True)

    segment_collection = []
    anomaly_collection = []
    for idx, group in stop_sequence.groupby(['route_id', 'direction_id', 'shape_id']):
        projections = _project_stop_on_route(idx, group)
        
        route_segments = _cut_route_into_segments(route, projections)

        segments_and_data = summ._assemble_route_segments_dataframe(route_segments, stop_ids, 
                                                                 stop_names, route_id,
                                                                 direction_id, shape_id,
                                                                 )
        segment_collection.append(segments_and_data)

        if flag_outliers:
            distances = projections['translation_distance'].to_numpy()
            anomalous_projections =  ug._flag_local_outliers(distances, stop_sequence, route_id,
                                                             direction_id, shape_id,
                                                             stop_ids, threshold)
            # TO DO: check consistency of following line... seems ok, though
            anomalous_projections['anom_type'] = projections.anomaly_type.to_numpy()
            
            anomaly_collection.append(anomalous_projections)

    cut_routes = pd.concat(segment_collection)

    if flag_outliers:
        anomalies = pd.concat(anomaly_collection)
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

