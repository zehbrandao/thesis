from . import parse
from . import utils_geo as ug



def get_hexagons_with_census_data(state, cities,
                                  hexagon_size, usecols,
                                  area_weighted_vars, query_data=True,
                                  save_query=True, path=None,
                                  pop_col='pop', pop_weighted_vars=None,
                                  output_epsg=31983,):
    """Gets the H3 hexagonal tesselation of a study area. It also
    inputs basic Census data into each hexagon — i.e. population
    and income.
    
    Parameters
    ----------
    state : str or int 
        State abbreviation (UF) or state code (according to IBGE records)
    cities : int, list
        IBGE's 7-digit identifier for the municipality(ies)
        in the study area
    hexagon_size : int
        One of H3's standar resolutions (see: h3geo.org)
    usecols : dict
        Dictionaty whose keys are census variables that are to be kept.
        Dict values are the corresponding new names for those variables
        in the returned DataFrame. Wherever col name is to be preserved
        dict value should be None.
    area_weighted_vars : str, list
        variables that are assumed to change as a function of area
    query_data : bool, def True
        Whether to query census data from Base dos Dados, if True, or
        to get file from disk, otherwise.
        
        Querying requires a Google Cloud Project to be billed. It is
        necessary to have a project of your own, even if empty, in
        order to retrieve data from Base dos Dados' database.
        (More on https://bit.ly/query_base_dos_dados)
        
    save_query : bool, def True
        Wheter to write query data to disk
    path : str, pathlib.Path object
        If query_data is True, place on disk where data is to be saved
        Otherwise, AND if save_query is True, refers to place on disk 
        where census data is to be stored.
    pop_col : str, def 'pop'
        name of column containing population count data
    pop_weighted_vars : str, list, def None
        variables assumed to be a function of population
    output_epsg: int or str, def 31983
        Desired CRS of the output gdf.
        Default refers to SIRGAS 2000 / UTM zone 23S
        
    Returns
    -------
    hexagons : GeoDataFrame
    """
    tract_data = parse._parse_census_data(usecols,
                                          query_data,
                                          save_query,
                                          path,)
    
    tract_shapes = parse._get_tract_shapes(state,
                                           cities,
                                           output_epsg,)
    
    tracts = parse._merge_tract_data_with_shape(tract_data,
                                                tract_shapes,)
    
    hexagons = ug.get_hexagons(tracts,
                               hexagon_size,
                               output_epsg,)
    
    hexagons = ug._spatially_transform_numerical_data(tracts,
                                                      hexagons,
                                                      area_weighted_vars,
                                                      pop_col,
                                                      pop_weighted_vars,)
    
    
    return hexagons
