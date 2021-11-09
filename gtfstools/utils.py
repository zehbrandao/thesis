"""General utility functions"""


def _fix_ids(roads, stops):
    """Relabels nodes so as to ensure that the IDs of POIs that are
    projected in the road netword do not conflict with those from osmnx.
    """
    no_nodes = len(roads)
    mapping = {old: new
               for old, new
               in zip(roads.nodes, range(1, no_nodes + 1))}
    roads = nx.relabel_nodes(roads, mapping)
    
    stop_ids = stops.stop_id
    alias_range = range(no_nodes + 1,
                        no_nodes + len(stop_ids) + 1)
    stop_id_corresp = {original: alias
                       for original, alias
                       in zip(stop_ids, alias_range)}
    
    
    return roads, stop_id_corresp