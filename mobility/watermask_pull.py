import glob
import numpy as np
import argparse
from scipy import stats
import os
import ee
import fiona
import rasterio
from IPython.display import HTML, display, Image
from matplotlib import pyplot as plt
import geemap as geemap
from natsort import natsorted
import requests
from rasterio.merge import merge

from landsat_fun import getPolygon 
from puller_fun import pull_watermasks 


# ee.Authenticate()
ee.Initialize()


polygon_path = '/Users/greenberg/Documents/PHD/Projects/Mobility/WaterMaskTesting/YellowTooBig.gpkg'
root = '/Users/greenberg/Documents/PHD/Projects/Mobility/WaterMaskTesting/monthly'
paths = pull_watermasks(polygon_path, root)


    if __name__ == '__main__':
        year = 1990
        polygon_path = '/Users/greenberg/Documents/PHD/Projects/Mobility/WaterMaskTesting/YellowTooBig.gpkg'
        # polygon_path = '/Users/greenberg/Documents/PHD/Projects/Mobility/WaterMaskTesting/YellowSmall.gpkg'
        root = '/Users/greenberg/Documents/PHD/Projects/Mobility/WaterMaskTesting/monthly'
        name = 'yellow'

        polygon_name = polygon_path.split('/')[-1].split('.')[0]
        polys = getPolygon(polygon_path, root, name, year)
        pull_months = None
        for i, poly in enumerate(polys):
            river, pull_months = pullYearMask(year, poly, root, name, i, pull_months)

        # Merge masks
        fps = glob.glob(os.path.join(root, '*chunk*.tif'))
        mosaics = []
        for fp in fps:
            ds = rasterio.open(fp)
            mosaics.append(ds)
        meta = ds.meta.copy()
        mosaic, out_trans = merge(mosaics)

        # Update the metadata
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        })

        out_path = os.path.join(
            root, 
            '{}_{}_{}.tif'
        )
        out_fp = out_path.format(name, year, 'full')
        with rasterio.open(out_fp, "w", **meta) as dest:
            dest.write(mosaic)

        for fp in fps:
            os.remove(fp)

        plt.imshow(mosaic[0,:,:])
        plt.show()
