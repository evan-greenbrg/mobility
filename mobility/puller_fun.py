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

from landsat_fun import getImageAllMonths
from landsat_fun import getImageSpecificMonths
from landsat_fun import getMonthImageAllPolys
from landsat_fun import getPolygon 
from channel_mask_fun import getWaterJones
from channel_mask_fun import getWaterZou
from channel_mask_fun import getRiverGRWL
from channel_mask_fun import getRiverMERIT
from channel_mask_fun import getRiverLARGEST
from channel_mask_fun import getMERITfeatures 
from download_fun import ee_export_image
from multiprocessing_fun import multiprocess


# ee.Authenticate()
ee.Initialize()
warnings.filterwarnings("ignore")

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


def pull_esa(polygon_path, out_root, export_images=False):

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
        year = fp.split('/')[-1].split('_')[1]
        ds = rasterio.open(fp)
        raw_image = ds.read(1)

        image, tf = rasterio.mask.mask(
            ds, [geom],
            crop=False, filled=False
        )

        # Threshold
        water = image.data[0, :, :] > 0
        if not np.sum(water):
            continue

        meta = ds.meta
        meta.update(
            width=water.shape[1],
            height=water.shape[0],
            count=1,
            dtype=rasterio.int8
        )

        images[year] = water 
        metas[year] = meta

        with rasterio.open(fp, "w", **meta) as dest:
            dest.write(water.astype(rasterio.uint8), 1)

    return images, metas


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
            out_image = out_image.astype('int64')
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
        images = getMonthImageAllPolys(year, polys, month)
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
                water = getWaterJones(ds).astype(int)
            elif mask_method =='Zou':
                water = getWaterZou(ds).astype(int)

            if len(np.unique(water)) == 1:
                continue

            if network_method == 'grwl':
                river = getRiverGRWL(water, ds.transform, bounds[i])
            elif network_method == 'merit':
                river = getRiverMERIT(water, ds.transform, network)
            elif network_method == 'largest':
                river = getRiverLARGEST(water)
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
        pull_months = getQuarters(min_water)

    for fp in fps:
        os.remove(fp)

    return pull_months


def get_pull_months(year, poly, root, name, 
                    bot=40, top=80, 
                    mask_method='Jones',
                    network_method='grwl', network=None):
    
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
#        ee_export_image(image, out, scale=30, file_per_band=False)

    multiprocess(tasks)

    fps = natsorted(glob.glob(os.path.join(root, '*_month.tif')))
    rivers = []
    for fp in fps:
        ds = rasterio.open(fp)

        if mask_method == 'Jones':
            water = getWaterJones(ds).astype(int)
        elif mask_method =='Zou':
            water = getWaterZou(ds).astype(int)

        if network_method == 'grwl':
            bound = image.geometry()
            river = getRiverGRWL(water, ds.transform, bound)
        elif network_method == 'merit':
            river = getRiverMERIT(water, ds.transform, network)
        elif network_method == 'largest':
            river = getRiverLARGEST(water)
        else:
            river = water

        rivers.append(river)

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


def getQuarters(min_water):
    quarters = []
    quarter = []
    for i, mo in enumerate(range(min_water, min_water + 12)):
        if mo > 12:
            mo -= 12

        if (i + 1) % 3:
            quarter.append(MONTHS[mo - 1])
        else:
            quarter.append(MONTHS[mo - 1])
            quarters.append(quarter)
            quarter = []

    return quarters


def pullYearMask(year, poly, root, name, chunk_i, block_i, pull_month,
                 export_images=False, mask_method='Jones',
                 network_method='grwl', network=None):

    out_path = os.path.join(
        root, 
        str(year),
        '{}_{}_{}.tif'
    )

    image = getImageSpecificMonths(year, pull_month, poly)

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
        water = getWaterJones(ds).astype(int)
    elif mask_method =='Zou':
        water = getWaterZou(ds).astype(int)

    if network_method == 'grwl':
        bound = image.geometry()
        river_im = getRiverGRWL(
            water, ds.transform, bound
        )
    elif network_method == 'merit':
        river_im = getRiverMERIT(
            water, ds.transform, network 
        )
    elif network_method == 'largest':
        river_im = getRiverLARGEST(water)
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


def pull_image_watermasks(polygon_path, root, export_images, 
                          mask_method='Jones', network_method='grwl', 
                          merit_path=None, period='bankfull'):

    years = [i for i in range(1985, 2020)]
    river_polys = getPolygon(polygon_path, root)
    river_paths = {}
    for river, polys in river_polys.items():
        print()
        print(river)

        # Get merit features
        if network_method == 'merit':
            print('finding networks')
            networks = getMERITfeatures(polys, merit_path)
        else:
            networks = [None for i in polys]

        # Make river dir 
        year_root = os.path.join(root, river)
        os.makedirs(year_root, exist_ok=True)

        # Pull yearly images
        pull_months = get_months(
            2018, polys, year_root, river, 
            mask_method=mask_method, 
            network_method=network_method,
            network=merit_path,
            period=period
        )

        # Iterate through all the years and all of the pull_months
        print(pull_months)
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
                        pullYearMask,
                        (
                            year, poly, year_root, 
                            river, poly_i, pull_i, 
                            pull_month, export_images,
                            mask_method, network_method, networks[poly_i]
                        )
                    ))
        multiprocess(tasks)

        print(pull_months)
        river_paths[river] = [] 
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

            river_paths[river].append(out_paths)

    return river_paths


def mosaic_images(year_root, year, river, pull_i, pattern):
    pattern_format = '*{}_block_{}*.tif'.format(pattern, pull_i)
    # Mosaic Masks
    fps = glob.glob(os.path.join(
        year_root, str(year), pattern_format
    ))
    if not fps:
        return None 

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

    out_root = os.path.join(
        year_root, 
        f'{pattern}'
    )
    os.makedirs(out_root, exist_ok=True)

    out_path = os.path.join(
        out_root, 
        '{}_{}_{}.tif'
    )
    out_fp = out_path.format(river, year, f'full_{pattern}_block_{pull_i}')

    with rasterio.open(out_fp, "w", **meta) as dest:
        dest.write(mosaic)

    for fp in fps:
        os.remove(fp)

    return out_fp


def get_paths(poly, root):
    # Get the rivers
    rivers = []
    polygon_name = poly.split('/')[-1].split('.')[0]
    with fiona.open(poly, layer=polygon_name) as layer:
        for feature in layer:
            rivers.append(feature['properties']['River'])

    river_paths = {}
    for river in rivers:
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

        river_paths[river] = out_paths

    return river_paths
