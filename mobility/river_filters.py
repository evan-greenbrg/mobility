import numpy as np
import argparse
import geopandas
import os
import ee
import ee.mapclient
import fiona
import rasterio
from skimage import measure, draw, morphology, feature, graph
from shapely.geometry import MultiLineString, LineString, Polygon
from shapely import ops
from IPython.display import HTML, display, Image
from matplotlib import pyplot as plt
import geemap as geemap
from skimage.graph import MCP 
from skimage.measure import label


def get_MERIT_features(polys, merit_path):
    network_features = []
    for i, poly in enumerate(polys):
        print('Poly: ', i)
        geom = np.array(poly.getInfo()['coordinates'])[0, :, :]

        xmin = geom[:, 0].min()
        xmax = geom[:, 0].max()
        ymin = geom[:, 1].min()
        ymax = geom[:, 1].max()

        gdf = geopandas.read_file(
            merit_path, 
            bbox=(xmin, ymin, xmax, ymax) 
        )

        network_features.append(gdf)

    return network_features


def get_river_MERIT(water, transform, network):

    lines = []
    for i, feature in network.iterrows():
        lines.append(feature['geometry'])
    
    if not lines:
        return [] 

    multi = ops.linemerge(MultiLineString(lines))
    rows = []
    cols = []
    if multi.geom_type == 'MultiLineString':
        for i, m in enumerate(multi):
            rs, cs = rasterio.transform.rowcol(
                transform, 
                m.xy[0], 
                m.xy[1] 
            )
            rows += rs
            cols += cs
    elif multi.geom_type == 'LineString':
        rs, cs = rasterio.transform.rowcol(
            transform, 
            multi.xy[0], 
            multi.xy[1] 
        )
        rows += rs
        cols += cs

    pos = np.empty((len(rows), 2))
    pos[:, 0] = rows
    pos[:, 1] = cols

    pos = np.delete(
        pos, 
        np.argwhere(
            pos[:, 0] >= water.shape[0]
        ),
        axis=0
    ).astype(int)

    pos = np.delete(
        pos, 
        np.argwhere(
            pos[:, 1] >= water.shape[1]
        ),
        axis=0
    ).astype(int)

    cl_raster = np.zeros(water.shape)
    cl_raster[pos[:, 0], pos[:, 1]] = 1
    cl_points = np.argwhere(cl_raster)

    # Extract a channel 
    not_water = 1 - np.copy(water)
    m = MCP(not_water)
    cost_array, _ = m.find_costs(
        cl_points, 
        max_cumulative_cost=30
    )
    return (cost_array == 0).astype(int)


def get_river_largest(water):
    labels = measure.label(water)
    
    if not labels.max():
        return water

     # assume at least 1 CC
    assert(labels.max() != 0)

    # Find largest connected component
    bins = np.bincount(labels.flat)[1:] 
    cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return cc.astype(int)


def get_river_GRWL(water, transform, bound):

    grwl = ee.FeatureCollection(
        "projects/sat-io/open-datasets/GRWL/water_vector_v01_01"
    )

    cl = grwl.filterBounds(bound)
    lines = []
    for feature in cl.getInfo()['features']:
        lines.append(LineString(feature['geometry']['coordinates']))
    
    if not lines:
        return [] 

    multi = ops.linemerge(MultiLineString(lines))
    rows = []
    cols = []
    if multi.geom_type == 'MultiLineString':
        for i, m in enumerate(multi):
            rs, cs = rasterio.transform.rowcol(
                transform, 
                m.xy[0], 
                m.xy[1] 
            )
            rows += rs
            cols += cs
    elif multi.geom_type == 'LineString':
        rs, cs = rasterio.transform.rowcol(
            transform, 
            multi.xy[0], 
            multi.xy[1] 
        )
        rows += rs
        cols += cs

    pos = np.empty((len(rows), 2))
    pos[:, 0] = rows
    pos[:, 1] = cols

    pos = np.delete(
        pos, 
        np.argwhere(
            pos[:, 0] >= water.shape[0]
        ),
        axis=0
    ).astype(int)

    pos = np.delete(
        pos, 
        np.argwhere(
            pos[:, 1] >= water.shape[1]
        ),
        axis=0
    ).astype(int)

    cl_raster = np.zeros(water.shape)
    cl_raster[pos[:, 0], pos[:, 1]] = 1
    cl_points = np.argwhere(cl_raster)

    # Extract a channel 
    not_water = 1 - np.copy(water)
    m = MCP(not_water)
    cost_array, _ = m.find_costs(
        cl_points, 
        max_cumulative_cost=30
    )
    return (cost_array == 0).astype(int)


