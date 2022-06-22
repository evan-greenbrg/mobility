import time
import fiona
import copy
import glob
import os
import ee
from scipy import stats
import fiona
import rasterio
import rasterio.mask
from rasterio.merge import merge
from skimage import measure, draw
import pandas
import numpy as np
import geemap as geemap
from natsort import natsorted
import warnings

from puller_helpers import get_polygon
from merit_helpers import get_MERIT_features 
from landsat_fun import get_month_image_all_polys
from watermask_methods import get_water_Jones
from watermask_methods import get_water_Zou
from river_filters import get_river_MERIT
from river_filters import get_river_GRWL
from river_filters import get_river_largest
from landsat_fun import get_image_specific_months
from landsat_fun import surface_water_image
from download import ee_export_image
from puller_helpers import mosaic_images 
from multiprocessing_fun import multiprocess


# ee.Authenticate()
ee.Initialize()
warnings.filterwarnings("ignore")

# Initialize multiprocessing
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08',
    '09', '10', '11', '12', ]


def pull_esa(polygon_path, out_root, export_images=False):

    years = [i for i in range(1985, 2020)]
    river_polys = get_polygon(polygon_path, out_root)
    river_paths = {}

    for river, polys in river_polys.items():

        print()
        print(river)
        river_root = out_root.format(river)
        os.makedirs(river_root, exist_ok=True)

        # Make image dir 
        year_root = os.path.join(river_root, 'temps')
        os.makedirs(year_root, exist_ok=True)

        out_paths = []

        # Get months that are average-bankfull flow
        pull_months = get_months(2018, polys[0], year_root, river)

        for j, year in enumerate(years):
            print(year)
            time.sleep(5)

            tasks = []
            for i, poly in enumerate(polys):
                tasks.append((
                    pull_rear_ESA,
                    (
                        year, poly, year_root, 
                        river, i, pull_months
                    )
                ))

            multiprocess(tasks)

            fps = glob.glob(os.path.join(year_root, '*mask*.tif'))
            if not fps:
                continue

            mosaics = []
            for fp in fps:
                ds = rasterio.open(fp)
                mosaics.append(ds)
            meta = ds.meta.copy()
            mosaic, out_trans = merge(mosaics)
            
            mosaic[mosaic < 2] = 0
            mosaic[mosaic > 1] = 1

            for fp in fps:
                os.remove(fp)

            if not np.sum(mosaic):
                continue

            # Update the metadata
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "dtype": rasterio.uint8
            })

            out_path = os.path.join(
                year_root, 
                '{}_{}_{}.tif'
            )
            out_fp = out_path.format(river, year, 'full')
            out_paths.append(out_fp)

            with rasterio.open(out_fp, "w", **meta) as dest:
                dest.write(mosaic.astype(rasterio.uint8))

        river_paths[river] = out_paths

    return river_paths


def get_months(year, polys, root, name, 
               bot=40, top=80,
               mask_method='Jones', 
               network_method='grwl',
               network=None, period='min'):

    out_path = os.path.join(
        root, 
        '{}_{}_{}.tif'
    )
    monthly_water = []
    for month in MONTHS:
        print(month)
        images = get_month_image_all_polys(year, polys, month)
        tasks = []
        bounds = []
        for i, image in enumerate(images):
            out = out_path.format(
                name, year, f'poly_{i}'
            )
            tasks.append((
                ee_export_image, 
                (image, out)
            ))

            bounds.append(image.geometry())

        multiprocess(tasks)
        fps = natsorted(glob.glob(os.path.join(root, '*_poly_*.tif')))

        water_pixels = 0
        for i, fp in enumerate(fps):
            ds = rasterio.open(fp)

            if mask_method == 'Jones':
                water = get_water_Jones(ds).astype(int)
            elif mask_method =='Zou':
                water = get_water_Zou(ds).astype(int)

            if len(np.unique(water)) == 1:
                continue

            if network_method == 'grwl':
                river = get_river_GRWL(water, ds.transform, bounds[i])
            elif network_method == 'merit':
                river = get_river_MERIT(water, ds.transform, network)
            elif network_method == 'largest':
                river = get_river_largest(water)
            else:
                river = water

            water_pixels += len(np.argwhere(river == 1))

        monthly_water.append(water_pixels)

    percentiles = np.array([
        stats.percentileofscore(monthly_water, i) 
        for i in monthly_water 
    ])
    if period == 'bankfull':
        ns, = np.where((percentiles > bot) & (percentiles < top))
        pull_months = [[MONTHS[n] for n in ns]]
    elif period == 'min':
        i = np.argmin(np.abs(percentiles - 25))
        pull_months = [[MONTHS[i]]]
    elif period == 'max':
        i = np.argmin(np.abs(percentiles - 85))
        pull_months = [[MONTHS[i]]]
    elif period == 'annual':
        pull_months = [MONTHS]
    elif period == 'quarterly':
        min_water = np.argmin(monthly_water)
        pull_months = get_quarters(min_water)

    for fp in fps:
        os.remove(fp)

    return pull_months


def pull_year_ESA(year, poly, root, name, chunk_i, pull_months):
    # outs
    out_path = os.path.join(
        root, 
        '{}_{}_{}.tif'
    )

    image = surface_water_image(year, poly)

    if not image.bandNames().getInfo():
        return None

    ee_export_image(
        image,
        filename=out_path.format(name, year, f'mask_{chunk_i}'),
        scale=30,
        file_per_band=False
    )

    return 1


def pull_year_mask(year, poly, root, name, chunk_i, block_i, pull_month,
                 export_images=False, mask_method='Jones',
                 network_method='grwl', network=None):

    out_path = os.path.join(
        root, 
        str(year),
        '{}_{}_{}.tif'
    )

    image = get_image_specific_months(year, pull_month, poly)

    if not image.bandNames().getInfo():
        return None

    out = out_path.format(
        name, 
        year, 
        f'image_block_{block_i}_poly_{chunk_i}'
    )
    status = ee_export_image(
        image,
        filename=out,
        scale=30,
        file_per_band=False
    )

    ds = rasterio.open(out)

    if mask_method == 'Jones':
        water = get_water_Jones(ds).astype(int)
    elif mask_method =='Zou':
        water = get_water_Zou(ds).astype(int)

    if network_method == 'grwl':
        bound = image.geometry()
        river_im = get_river_GRWL(
            water, ds.transform, bound
        )
    elif network_method == 'merit':
        river_im = get_river_MERIT(
            water, ds.transform, network 
        )
    elif network_method == 'largest':
        river_im = get_river_largest(water)
    else:
        river_im = water

    if not export_images:
        os.remove(out)
    if not len(river_im):
        return None

    meta = ds.meta
    meta.update({
        'count': 1,
        'dtype': rasterio.uint8
    })

    out = out_path.format(
        name, 
        year, 
        f'mask_block_{block_i}_poly_{chunk_i}'
    )

    with rasterio.open(out, 'w', **meta) as dst:
        dst.write(river_im.astype(rasterio.uint8), 1)

    return river_im


def pull_watermasks(polygon_path, root, river, export_images, 
                          mask_method='Jones', network_method='grwl', 
                          merit_path=None, period='bankfull'):

    # Get small enough polygons
    years = [i for i in range(1985, 2020)]
    polys = get_polygon(polygon_path, root)

    # If I'm using the dense river network file
    if network_method == 'merit':
        print('finding networks')
        networks = get_MERIT_features(polys, merit_path)
    else:
        networks = [None for i in polys]

    # Make river dir 
    year_root = os.path.join(root, river)
    os.makedirs(year_root, exist_ok=True)

    # Get blocks of months to pull 
    pull_months = get_months(
        2018, polys, year_root, river, 
        mask_method=mask_method, 
        network_method=network_method,
        network=merit_path,
        period=period
    )
    print(pull_months)

    # Iterate through all the years and all of the pull_months
    tasks = []
    for year_i, year in enumerate(years):
        os.makedirs(
            os.path.join(
                year_root, str(year),
            ), exist_ok=True
        )
        for pull_i, pull_month in enumerate(pull_months):
            for poly_i, poly in enumerate(polys):
                tasks.append((
                    pull_year_mask,
                    (
                        year, poly, year_root, 
                        river, poly_i, pull_i, 
                        pull_month, export_images,
                        mask_method, network_method, networks[poly_i]
                    )
                ))
    multiprocess(tasks)

    river_paths = []
    for pull_i, pull_month in enumerate(pull_months):
        out_paths = []
        for year_i, year in enumerate(years):
            # Images
            if export_images:
                pattern = 'image'
                out_fp = mosaic_images(
                    year_root, year, river, pull_i, pattern
                )

            # Mask
            pattern = 'mask'
            out_fp = mosaic_images(
                year_root, year, river, pull_i, pattern
            )
            if not out_fp:
                continue

            out_paths.append(out_fp)

        river_paths.append(out_paths)

    return river_paths


def get_paths(poly, root, river):
    # Get the rivers
    fps = glob.glob(os.path.join(root, river, 'mask', '*'))
    blocks = {}
    for fp in fps:
        block = fp.split('_')[-1].split('.')[0]
        if not blocks.get(block):
            blocks[block] = [fp]
        if blocks.get(block):
            blocks[block].append(fp)

    out_paths = []
    for block in sorted(blocks.keys()):
        out_paths.append(blocks[block])

    return out_paths
