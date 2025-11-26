import re

import branca
import folium

import mapclassify as mc
import seaborn as sns

from .utils_time import _slice_gtfs_by_direction_and_time_window
from .utils_parsing import _get_shapes_into_route_summary


def _get_color_thresholds(data, variable, method, k):
    try:
        classifier = getattr(mc, method)
    except ValueError:
        print(
            "Name of a choropleth classification scheme from mapclassify module. \n"
            "Supported are all schemes provided by mapclassify (e.g.'BoxPlot', \n"
            "'EqualInterval', 'FisherJenks', 'FisherJenksSampled', 'HeadTailBreaks', \n"
            "'JenksCaspall', 'JenksCaspallForced', 'JenksCaspallSampled', 'MaxP', \n"
            "'MaximumBreaks', 'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean','UserDefined'"
        )
    
    classes = classifier(data[variable], k).get_legend_classes()
    expression = r'(\d*\.\d*),\s+(\d*\.\d*)'
    
    breaks = []
    for class_ in classes:
        class_bounds = re.search(expression, class_).groups()
        class_bounds = [float(bound) for bound in class_bounds]
        breaks.extend(class_bounds)
        
    # MEMENTO: from Python 3.7, the regular dict is guaranteed
    # to be ordered across all implementations.
    breaks = list(dict.fromkeys(breaks)) 
    
    
    return breaks
    
        
def _get_branca_colormap(data, variable, method=None, k=None, cmap='YlOrRd', linear=True):
    rgb_tuples = sns.color_palette(cmap)
    cmap = branca.colormap.LinearColormap(colors=rgb_tuples,
                                          vmin=data[variable].min(),
                                          vmax=data[variable].max(),)
    breaks = _get_color_thresholds(data, variable, method, k)
    if not linear:
        cmap = cmap.to_step(index=breaks,          # TO DO: verify if the rightmost bound is necesary
                            round_method='int',) # MEMENTO: rounds up to the nearest order-of-magnitude integer.
        
        
    return cmap, breaks


def _get_folium_map_object(data, tiles):
    # to plot the data on a folium map, we need to convert to a
    #Geographic coordinate system with the wgs84 datum (EPSG: 4326).

    minx, miny, maxx, maxy = data.geometry.total_bounds
    lat = miny + (maxy - miny)/2
    lon = minx + (maxx - minx)/2

    map_ = folium.Map(location=[lat, lon], # MEMENTO: it is NOT GeoJSON conformant
                      tiles=tiles,
                      zoom_start=12,)
    
    
    return map_


def plot_gtfs_data(gtfs_data, variable, window,
                   direction=None, method=None, 
                   k=None, cmap='YlOrRd',
                   linear=True, tiles='Stamen Terrain'):
    
    route_summary, stop_times, shapes = gtfs_data
    operations_within_time_window = _slice_gtfs_by_direction_and_time_window(route_summary,
                                                                             window,
                                                                             direction,)
    operations_within_time_window = _get_shapes_into_route_summary(stop_times,
                                                                   operations_within_time_window,
                                                                   shapes,)
    
    operations_within_time_window.set_index('route_id',
                                            inplace=True,)
        
        
    map_ = _get_folium_map_object(operations_within_time_window,
                                  tiles)
    cmap, breaks = _get_branca_colormap(operations_within_time_window,
                                        variable,
                                        method,
                                        k,
                                        cmap,
                                        linear,)
    
    operations_within_time_window.to_crs(epsg=4326,
                                         inplace=True) # folium takes data in wgs84 datum for input
    
    def _style_function(feature,
                        operations_within_time_window,
                        color_corresp, cmap):
        # TO DO: thickness dict
        style_parameters = {
            'color': cmap(color_corresp[feature['id']]),
            'fillOpacity': 0.5,
            'weight': 3,
        }
        return style_parameters
    
    color_corresp = operations_within_time_window[variable].to_dict()
    
    
    folium.GeoJson(
        operations_within_time_window.to_json(),
        style_function=lambda feature: _style_function(feature,
                                                       operations_within_time_window,
                                                       color_corresp,
                                                       cmap), 
    ).add_to(map_)
    
    cmap.caption = variable
    cmap.add_to(map_)
    
    folium.LayerControl(collapsed=False).add_to(map_)
    
    
    return map_