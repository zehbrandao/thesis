import pathlib

import pandas as pd
import basedosdados as bd

from geobr import read_census_tract
from getpass import getpass


##############################################################################

def _parse_to_str(df, col):
    """Takes a database with a numerical column and 
    turns it into string.
    """
    df = df.astype({col: 'int64'}) # This script will handle big numbers
    df = df.astype({col: str})
    
    return df


def _query_census_data():
    """Queries basic 2010 Census data from Base dos Dados,
    converts tract IDs to str and sets them as index.
    """
    project_id = getpass("Type your Google Cloud Project ID:")
    tract_data = bd.read_table(dataset_id='br_ibge_censo_demografico',
                               table_id='setor_censitario_basico_2010',
                               billing_project_id=project_id,)
    
    # Census tract IDs are read in as int and it's better to use str
    tract_data = _parse_to_str(tract_data, 'id_setor_censitario')
    tract_data.set_index('id_setor_censitario', inplace=True)
    
    
    return tract_data


def _write_data(df, path):
    if path:
        print('Writing query to informed path...')
        df.to_csv(path)
        print('Done!')
    else:
        print('path is None, writing query to cwd...')
        df.to_csv('setor_censitario_basico_2010.csv')
        print('Done!')


def _get_census_data(query_data, save_query, path):
    """Obtains census data, whether from a query to Base dos Dados'
    repository of from disk. If data comes from query, allows writing
    on disk.
    """
    if query_data:
        tract_data = _query_census_data()
    else:
        tract_data = pd.read_csv(path, index_col=0)
        tract_data.index = tract_data.index.astype(str)
    
    if save_query:
        _write_data(tract_data, path)
    
    
    return tract_data
    
    
def _parse_census_data(usecols, query_data, save_query, path):
    """Takes populational data from the 2010 Census as provided
    by Base dos Dados (refer to https://basedosdados.org/). The
    query returns a CSV for all census tracts of Brazil, which is
    then sliced.
    
    Querying requires a Google Cloud Project to be billed. It is
    necessary to have a project of your own, even if empty, in order
    to retrieve data from Base dos Dados' database.
    (More on https://bit.ly/query_base_dos_dados)
    
    Refer to https://bit.ly/ibge_2010_census_repo for downloading 
    census data manually and for access to the survey documentation.
    
    Parameters
    ----------
    usecols : dict
        Dictionaty whose keys are census variables that are to be kept.
        Dict values are the corresponding new names for those variables
        in the returned DataFrame. Wherever col name is to be preserved
        dict value should be None.
    query_data : bool, def True
        Whether to query census data from Base dos Dados, if True, or
        to get file from disk, otherwise.
    save_query : bool, def True
        Wheter to write query data to disk
    path : str, pathlib.Path object
        If query_data is True, place on disk where data is to be saved
        Otherwise, AND if save_query is True, refers to place on disk 
        where census data is to be stored.
        
    Returns
    -------
    tract_data : DataFrame
    """
    tract_data = _get_census_data(query_data,
                                  save_query,
                                  path,)
    
    # All the following assumes data is structured in the
    # same way as returned by _query_census_data()
    to_keep = list(usecols.keys())
    tract_data = tract_data.reindex(columns=to_keep)
    
    names = {old: new for old,new in usecols.items() if new is not None}
    try:
        tract_data.rename(columns=names, inplace=True)
    except:
        pass
    
    
    return tract_data


def _get_tract_shapes(state, cities, output_epsg):
    """Gets census tracts from study area, while dropping all
    columns that are not relevant in the Thesis context.
    
    Parameters
    -----------
    state : str or int 
        State abbreviation (UF) or the state's IBGE code
    cities: list or str
        Municipalities in the study area
        
        TO DO: allow for int as well
    output_epsg : int
        EPSG code desired for the output
    
    Returns
    -------
    tracts : gdf
        GeoDataFrame containing two columns: 
        one with the IBGE codes for each tract 
        and another with the corresponding geometries
    """
    # TO DO: consider if it's necessary to allow handling 
    # of cities in different states
    tract_shapes = read_census_tract(state)
    
    # The above function returns IDs as float, but
    # I prefer to work with them as str, hence the
    # type conversion below.
    for each in ['code_tract', 'code_muni']:
        tract_shapes = _parse_to_str(tract_shapes, each)
        
    tract_shapes.to_crs(epsg=output_epsg, inplace=True)
    
    try:
        mask = tract_shapes.code_muni.isin(cities)
    except:
        mask = tract_shapes.code_muni==cities
    
    tract_shapes = tract_shapes.loc[mask]
    
    
    return tract_shapes[['code_tract', 'geometry']]


def _merge_tract_data_with_shape(tract_data, tract_shapes):
    tract_shapes.set_index('code_tract', inplace=True)
    tracts = tract_shapes.merge(tract_data,
                                how='left',
                                left_index=True,
                                right_index=True,)
    
    
    return tracts