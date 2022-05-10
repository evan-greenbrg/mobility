import time
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

from landsat_fun import getImageAllMonths
from landsat_fun import getImageSpecificMonths
from landsat_fun import getPolygon 
from channel_mask_fun import getWater
from channel_mask_fun import getRiver
from channel_mask_fun import fillHoles
from download_fun import ee_export_image
from multiprocessing_fun import multiprocess


# ee.Authenticate()
ee.Initialize()

# Initialize multiprocessing
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08',
    '09', '10', '11', '12', ]


def getSurfaceWater(year, polygon):
    sw = ee.ImageCollection("JRC/GSW1_3/YearlyHistory")

    begin = str(year) + f'-01' + '-01'
    end = str(year) + f'-12' + f'-31'

    return sw.filterDate(
        begin, end
    ).median().clip(
        polygon
    )


def get_bound(polygon_path, river):
    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            geom_river = feature['properties']['River']

            if geom_river != river:
                continue

            return ee.geometry.Geometry(feature['geometry'])


def pull_esa(polygon_path, out_root):

    years = [i for i in range(1985, 2020)]
    river_polys = getPolygon(polygon_path, out_root)
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
        pull_months = get_pull_months(2018, polys[0], year_root, river)

        for j, year in enumerate(years):
            print(year)
            time.sleep(5)

            tasks = []
            for i, poly in enumerate(polys):
                tasks.append((
                    pullYearESA,
                    (year, poly, year_root, river, i, pull_months)
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
            print(meta)
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


def clean_esa(poly, river, fps):
    grwl = ee.FeatureCollection(
        "projects/sat-io/open-datasets/GRWL/water_vector_v01_01"
    )

    polygon_name = poly.split('/')[-1].split('.')[0]
    with fiona.open(poly, layer=polygon_name) as layer:
        for feature in layer:
            geom_river = feature['properties']['River']
            bound = ee.geometry.Geometry(feature['geometry'])

            if geom_river != river:
                continue
            geom = feature['geometry']

    images = {}
    metas = {}
    for fp in fps:
        year = fp.split('_')[-1].split('.')[0]
        ds = rasterio.open(fp)
        raw_image = ds.read(1)

        image, tf = rasterio.mask.mask(
            ds, [geom],
            crop=False, filled=False
        )

        # Threshold
        water = image.data[0, :, :] > 0
        print(water.shape)

        meta = ds.meta
        meta.update(
            width=water.shape[1],
            height=water.shape[0],
            count=1,
            dtype=rasterio.int8
        )

        with rasterio.open(fp, "w", **meta) as dest:
            dest.write(water.astype(rasterio.uint8), 1)

    return 1 

#                images[year] = water 
#                metas[year] = water 
#
#    return images, metas


def create_mask(images):
    # Create Complete mask
    masks = {}
    for river, years in images.items():
        if not len(years):
            continue
        for year, im in years.items():
            im = im.astype(int)
            if int(year) == 1985:
                masks[river] = im
            else:
                masks[river] = np.add(masks[river], im)
        masks[river][masks[river] > 0] = 1

    return masks


def create_mask_shape(polygon_path, river, fps):
    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            geom_river = feature['properties']['River']

            if geom_river != river:
                continue

            geom = feature['geometry']

            image = fps[0]
            ds = rasterio.open(image)
            out_image, out_transform = rasterio.mask.mask(
                ds, [geom],
                crop=False, filled=False
            )
            out_image += 11
            out_image[np.where(out_image < 10)] = 0
            out_image[np.where(out_image > 10)] = 1

            return out_image[0, :, :]


def clean_channel_belt(mask, thresh=100):
    labels = measure.label(mask)
    # assume at least 1 CC
    # Find largest connected component
    channel = labels == np.argmax(np.bincount(labels.flat)[1:])+1
#        channel = fillHoles(channel, thresh)

    labels = measure.label(channel)
    # assume at least 1 CC
    # Find largest connected component
    clean_mask = labels == np.argmax(
        np.bincount(labels.flat)[1:]
    ) + 1

    return clean_mask


def fillHoles(mask, thresh):
    # Find contours
    contours = measure.find_contours(mask, 0.8)
    # Display the image and plot all contours found
    for contour in contours:
        # Get polygon
        poly = draw.polygon(contour[:, 0], contour[:, 1])
        if (len(poly[0]) == 0) or (len(poly[1]) == 0):
            continue
        area = (
            (poly[0].max() - poly[0].min() + 1)
            * (poly[1].max() - poly[1].min() + 1)
        )
        # Filter by size
        if area < thresh:
            draw.set_color(
                mask,
                poly,
                True
            )

    return mask


def crop_raster(raster, channel_belt):
    raster[np.where(channel_belt != 1)] = 0

    return raster


def filter_images(images, mask, thresh=.2):
    A = np.sum(mask)
    images_clean = {}
    images_keep = copy.deepcopy(images)
    for year, image in images.items():
        frac = np.sum(image[np.where(mask)]) / A
        if frac < thresh:
            images_keep.pop(year)

    return images_keep


def get_pull_months(year, poly, root, name, bot=40, top=80):
    
    grwl = ee.FeatureCollection(
        "projects/sat-io/open-datasets/GRWL/water_vector_v01_01"
    )
    out_path = os.path.join(
        root, 
        '{}_{}_{}.tif'
    )
    # Pull monthly images
    images = getImageAllMonths(year, poly)

    tasks = []
    for i, image in enumerate(images):
        out = out_path.format(
            name, year, f'{MONTHS[i]}_month'
        )
        tasks.append((
            ee_export_image, 
            (image, out)
        ))

    multiprocess(tasks)

    bound = image.geometry()
# Get river masks
    fps = natsorted(glob.glob(os.path.join(root, '*_month.tif')))
    rivers = []
    for fp in fps:
        ds = rasterio.open(fp)
        water = getWater(ds).astype(int)
        river = getRiver(water, ds.transform, bound, grwl)
        rivers.append(fillHoles(river, 12))

    # Get river_props
    water_pixels = []
    for river in rivers:
        water_pixels.append(len(np.argwhere(river == 1)))
    water_pixels = np.array(water_pixels)

    percentiles = np.array([
        stats.percentileofscore(water_pixels, i) 
        for i in water_pixels
    ])
    ns, = np.where((percentiles > bot) & (percentiles < top))
    pull_months = [MONTHS[n] for n in ns]

    for fp in fps:
        os.remove(fp)

    return pull_months


def pullYearESA(year, poly, root, name, chunk_i, pull_months):
    # outs
    out_path = os.path.join(
        root, 
        '{}_{}_{}.tif'
    )

    image = getSurfaceWater(year, poly)

    if not image.bandNames().getInfo():
        return None

    ee_export_image(
        image,
        filename=out_path.format(name, year, f'mask_{chunk_i}'),
        scale=30,
        file_per_band=False
    )

    return 1


def pullYearMask(year, poly, root, name, chunk_i, pull_months):

    # constants
    grwl = ee.FeatureCollection(
        "projects/sat-io/open-datasets/GRWL/water_vector_v01_01"
    )

    # outs
    out_path = os.path.join(
        root, 
        '{}_{}_{}.tif'
    )

    image = getImageSpecificMonths(year, pull_months, poly)

    if not image.bandNames().getInfo():
        return None

    bound = image.geometry()
    ee_export_image(
        image,
        filename=out_path.format(name, year, f'image_{chunk_i}'),
        scale=30,
        file_per_band=False
    )
    ds = rasterio.open(out_path.format(name, year, f'image_{chunk_i}'))
    water = getWater(ds).astype(int)
    river_im = getRiver(water, ds.transform, bound, grwl)

    if not len(river_im):
        os.remove(out_path.format(name, year, f'image_{chunk_i}'))
        return None

    river_im = fillHoles(river_im, 12)
    os.remove(out_path.format(name, year, f'image_{chunk_i}'))

    meta = ds.meta
    meta.update({
        'count': 1,
        'dtype': rasterio.uint8
    })

    with rasterio.open(
        out_path.format(name, year, f'mask_{chunk_i}'), 'w', **meta
    ) as dst:
        dst.write(river_im.astype(rasterio.uint8), 1)

    fps = glob.glob

    return river_im


def pull_watermasks(polygon_path, root):
    years = [i for i in range(1985, 2020)]
    river_polys = getPolygon(polygon_path, root)
    river_paths = {}
    for river, polys in river_polys.items():
        print()
        print(river)
        # Make river dir 
        river_root = root.format(river)
        os.makedirs(river_root, exist_ok=True)

        # Make image dir 
        year_root = os.path.join(river_root, 'temps')
        os.makedirs(year_root, exist_ok=True)

        # Pull yearly images
        start_time = time.time()
        out_paths = []

        pull_months = get_pull_months(2018, polys[0], year_root, river)
#        pull_months = None
        for j, year in enumerate(years):
            print(j)
            print(year)
            time.sleep(15)
            tasks = []
            for i, poly in enumerate(polys):
                tasks.append((
                    pullYearMask,
                    (year, poly, year_root, river, i, pull_months)
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

            # Update the metadata
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
            })

            out_path = os.path.join(
                year_root, 
                '{}_{}_{}.tif'
            )
            out_fp = out_path.format(river, year, 'full')
            out_paths.append(out_fp)

            with rasterio.open(out_fp, "w", **meta) as dest:
                dest.write(mosaic)

            for fp in fps:
                os.remove(fp)

        river_paths[river] = out_paths

    return river_paths
