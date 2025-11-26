import pathlib
import warnings

import geopandas as gpd
import pandas as pd
import partridge as ptg


def load_feed(path, busy_date=True):
    """Get gtfs data using partridge.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Path to gtfs folder, which can be (optionally) zipped
    
    Returns
    -------
        feed object
    """
    if busy_date:
        _, service_ids = ptg.read_busiest_date(path)
        view = {'trips.txt': {'service_id': service_ids}}
    else:
        view = {}
        
    feed = ptg.load_geo_feed(path, view)
    
    incomplete_stop_times = (
        feed.stop_times.departure_time.notnull().sum() < len(feed.stop_times)
    )
    
    if incomplete_stop_times:
        print('Timetable is incomplete.')
    
    if not feed.frequencies.empty:
        print('This is a frequency-based GTFS.')
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        if feed.shapes.empty:
            print('feed.shapes is empty')

        required_shapes = feed.trips.route_id.unique()

        if len(feed.shapes) < len(required_shapes):
            print('Feed contains less shapes than routes')
    
    
    return feed


def _handle_zipped_files(path):
    """ Detects if the path for the geospatial file is zipped 
    and parses it into a format that gpd.read_file() understands.
    """
    try:
        if path.suffix == '.zip':
            return (r'zip://' + path.as_posix())
        else:
            return path
    except AttributeError:
        if path.split('.')[-1] == 'zip':
            return (r'zip://' + pathlib.Path(path).as_posix())
        else:
            return path


def read_route_shapes(path):
    """Takes either a full raw string path os a pathlib's pure windows
    path and uses it to return a shapefile. It also makes the necessary
    adjustments to read shapefiles compressed into a .zip file.
    """    
    path = _handle_zipped_files(path)
            
            
    return gpd.read_file(path)


def _get_shapes_into_route_summary(stop_times, route_summary, route_shapes):
    """
    Makes the correspondence between routes and their shapes
    in a way that specifically handles the structure of the DataFrame
    returned by .gtfs.summarize_trips()
    
    # TO DO: assert this function is pertinent. Maybe describe data
    structure for 'route_shapes' explicitly.
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
