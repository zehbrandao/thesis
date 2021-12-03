import h3

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import mapping, Polygon, Point


def _get_H3_labels(hexagon_size, boundaries):
    """Finds all H3 hex labels of a given apperture that 
    lie within the study area.
    
    Parameters
    ----------
    hexagon_size: int
        Hexagon resolution (see h3geo.org, for a table with all resolutions)
    boundaries: list
        Collection of polygons that make up the study area
        
    Returns
    -------
    labels: DataFrame
        A single column pandas DataFrame, where each row contains
        the label of a H3 hexagon
    """
    labels = []
    for boundary in boundaries:
        temp = mapping(boundary)
        temp['coordinates'] = [
            
            [[lon,lat] for lon,lat in shape] for shape in temp['coordinates']
        ]
        
        # Memento: GeoJSON is in lon,lat format
        labels.extend(
            h3.polyfill(temp, hexagon_size, geo_json_conformant=True)
        )
        
    labels = pd.DataFrame(labels, columns=[f'hex_{hexagon_size}'])
       
    no_labels_with_duplicates = len(labels)
        
    labels.drop_duplicates(inplace=True)
        
    print('Sanity check:\n',
          f"{len(labels)} H3 labels, with resolution {hexagon_size}, ",
          f"after dropping {no_labels_with_duplicates-len(labels)} duplicates")
        
        
    return labels



def get_hexagons(gdf, hexagon_size, output_epsg):
    """Gets H3 hexagon labels that poplate the study area represented
    by the collection of geometries that is served as input.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Collection of spatial units that togetehr constitute
        the study area (e.g., census tracts)
    hexagon_size : int
        Hexagon resolution size (see: h3geo.org for the default list)
    output_epsg : int or str, def 31983
        Desired CRS of the output gdf.
        Default refers to SIRGAS 2000 / UTM zone 23S
        
    Returns
    -------
    hexagons : GeoDataFrame
    """
    # This functions works with EPSG:4326 because it's H3's base CRS
    # Convertion to output_epsg is made at the end.
    boundaries = gdf.to_crs(epsg=4326).geometry.unary_union
    
    # The following block is necessary because if polygons in gdf constitute
    # a single continuous footprint, the 'boundaries' variable receives a
    # shapely Polygon object, while '_get_H3_labels()' requires an iterable.
    try:
        labels = _get_H3_labels(hexagon_size, boundaries)
    except:
        boundaries = [boundaries]
        labels = _get_H3_labels(hexagon_size, boundaries)
    
    labels['hex_vertices_coords'] = labels[f'hex_{hexagon_size}'].map(
        # Memento (again): GeoJSON is in lon,lat format
        lambda x: h3.h3_to_geo_boundary(x, geo_json=True)
                                                                     ) 
    labels['geometry'] = labels.hex_vertices_coords.map(lambda x: Polygon(x))
    labels.drop(columns='hex_vertices_coords', inplace=True)
    
    hexagons = gpd.GeoDataFrame(labels, crs='EPSG:4326', geometry='geometry')
    hexagons.set_index(f'hex_{hexagon_size}', inplace=True)
    
    
    return hexagons.to_crs(epsg=output_epsg)


def _area_weighted_transformations(source, destination, area_weighted_vars):
    """Transfers data between spatial tesselations. Deals only with
    attributes that may be assumed as being a funtion of area.
    
    Parameters
    ----------
    -> refer to function 'spatially_transform_numerical_data'
    for parameter definitions
    
    Returns
    -------
    area_weighted_vars : GeoDataFrame
        Overlay of source and destination geometries. Each polygon
        contains the area corrected attribute (where applicable).
    """
    # Indexes are reset lest the labelling information is
    # lost after overlay
    source['source_area_m2'] = source.geometry.area
    from_ = source.reset_index()
    
    to = destination.reset_index()
    
    intersection = gpd.overlay(from_,
                               to,
                               how='intersection',
                               )
    # TO DO: the abve might be best in a function of its own
    
    intersectioned_areas_m2 = intersection.geometry.area
    area_weights = intersectioned_areas_m2 / intersection.source_area_m2
    
    for each in area_weighted_vars:
        intersection.loc[:,each] *= area_weights.to_numpy()
    
    intersection.drop(columns='source_area_m2', inplace=True)
    
    
    return intersection


def _population_based_transformations(gdf, pop_weighted_vars,
                                      destination_labels, pop_col):
    """Transfers data between spatial tesselations. Deals only with
    attributes are better assumed as being a funtion of population.
    
    Parameters
    ----------
    gdf : GeoDataFrame
    -> refer to function '_spatially_transform_numerical_data' 
    for remaining parameters
    
    Returns
    -------
    data : GeoDataFrame
    """
    
    pop = gdf.groupby(destination_labels)[pop_col].sum()
    pop.name = 'base_pop'
    
    data = gdf.merge(pop,
                     how='left',
                     left_on=destination_labels,
                     right_index=True,)
     
    pop_weights = data[pop_col] / data.base_pop
    #pop_weights = np.reshape(pop_weights.to_numpy(), (-1,1))
    
    for each in pop_weighted_vars:
        data.loc[:,each] *= pop_weights.to_numpy()
    
    
    return data.drop(columns='base_pop')


def _get_population_weighted_centroids(gdf, pop_col,
                                       destination_labels):
    """Adjusts centroid position based on population distribution.
    Centroids move towards areas with higher population counts.
    
    Parameters
    ----------
    gdf : GeoDataFrame
    -> refer to function 'spatially_transform_numerical_data'
    for remaining parameter definitions
    
    Returns
    --------
    corrected_coords : DataFrame
        Centroid locations indexed by destination
        geometry label (or ID)
    """
    data = gdf.copy()
    data['centres'] = [shape.centroid for shape in data.geometry]
    data['x'] = [centroid.x for centroid in data.centres]
    data['y'] = [centroid.y for centroid in data.centres]
    
    data.sort_values(destination_labels, inplace=True)
    
    # The following two lines are meant to prevent errors in
    # shapes with zero population. It is just a lame trick.
    #
    # TO DO: verify if this solution is actually robust
    data.replace({pop_col: {0: 1}}, inplace=True)
    data.fillna({pop_col: 1}, inplace=True)
    
    data['x'] = data['x'] * data[pop_col]
    data['y'] = data['y'] * data[pop_col]
    corrected_coords = data.groupby(destination_labels)[['x','y', pop_col]].sum()
    corrected_coords['x'] /= corrected_coords[pop_col]
    corrected_coords['y'] /= corrected_coords[pop_col]
    
    # I first thought of creating point geometries,
    # but GeoPackage does not handle files with more
    # than one column containing geometry data
    epsg = data.crs.to_epsg()
    new_names = {'x': f'x_epsg{epsg}', 'y': f'y_epsg{epsg}'}
    corrected_coords.rename(columns=new_names, inplace=True)
    
    
    return corrected_coords[[f'x_epsg{epsg}', f'y_epsg{epsg}']]    

    
def _spatially_transform_numerical_data(source, destination,
                                        area_weighted_vars, pop_col,
                                        pop_weighted_vars,):
    """Inputs data from source geometries into destination geometries,
    while performing the appropriate weightings and transformations.
    
    It also computes the centroid of the geometries in the returned
    GeoDataFrame. Centroids are weighted by population, i.e., they
    are biased towards largest populational concentrations.
    
    Parameters
    ----------
    source : GeoDataFrame
    destination : GeoDataFrame
    area_weighted_vars : str, list
        variables that are assumed to change as a function of area
    pop_col : str, def 'pop'
        name of column containing population count data
    pop_weighted_vars : str, list, def None
        variables assumed to be a function of population
        
    Returns
    -------
    destination : GeoDataFrame
    """
    transformed_data = _area_weighted_transformations(source,
                                                      destination,
                                                      area_weighted_vars,)
    destination_labels = destination.index.name
    if pop_weighted_vars is not None:
        transformed_data = _population_based_transformations(transformed_data,
                                                             pop_weighted_vars,
                                                             destination_labels,
                                                             pop_col,)
    
    centroids = _get_population_weighted_centroids(transformed_data,
                                                   pop_col,
                                                   destination_labels,)
    
    transformed_data = transformed_data.groupby(destination_labels).sum()
    
    destination = destination.merge(transformed_data,
                                    how='left',
                                    left_index=True,
                                    right_index=True,)
    
    destination = destination.merge(centroids,
                                    how='left',
                                    left_index=True,
                                    right_index=True,)
    
    
    return destination