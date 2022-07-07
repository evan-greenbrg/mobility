import glob
import os
import ee
import fiona
import rasterio
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.ops import split
import rasterio.mask
from rasterio.merge import merge
import numpy as np
import warnings

from landsat import get_image
from landsat import request_params


# ee.Authenticate()
ee.Initialize()
warnings.filterwarnings("ignore")

# Initialize multiprocessing
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08',
          '09', '10', '11', '12', ]


def split_polygon(shape, nx, ny):
    minx, miny, maxx, maxy = shape.bounds
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny

    minx, miny, maxx, maxy = shape.bounds
    dx = (maxx - minx) / nx  # width of a small part
    dy = (maxy - miny) / ny  # height of a small part

    horizontal_splitters = [
        LineString([(minx, miny + i*dy), (maxx, miny + i*dy)])
        for i in range(ny)
    ]
    vertical_splitters = [
        LineString([(minx + i*dx, miny), (minx + i*dx, maxy)])
        for i in range(nx)
    ]

    splitters = horizontal_splitters + vertical_splitters
    result = shape

    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))

    return result


def get_polygon(polygon_path, root, year=2018):

    out_path = os.path.join(
        root,
        '{}_{}_{}.tif'.format('temp', year, 'river')
    )

    # Load initial polygon
    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            print(feature)
            geom = feature['geometry']
            poly_shape = Polygon(geom['coordinates'][0])
            poly = ee.Geometry.Polygon(geom['coordinates'])

            image = get_image(year, poly)

            params = request_params(out_path, 30, image)

            outcomes = []
            try:
                url = image.getDownloadURL(params)
                outcomes.append(True)
                return [poly]

            except:
                outcomes.append(False)

            nx = 2
            ny = 2
            while False in outcomes:
                shapes = [
                    i
                    for i in split_polygon(poly_shape, nx, ny)
                ]
                nx += 1
                ny += 1
                outcomes = []

                for shape in shapes:
                    coordinates = np.swapaxes(
                        np.array(shape.exterior.xy), 0, 1
                    ).tolist()
                    poly = ee.Geometry.Polygon(coordinates)

                    image = get_image(year, poly)
                    params = request_params(out_path, 30, image)

                    try:
                        url = image.getDownloadURL(params)
                        outcomes.append(True)
                    except:
                        outcomes.append(False)

            shapes = [
                i
                for i in split_polygon(poly_shape, nx+1, ny+1)
            ]
            polys = []
            for shape in shapes:
                coordinates = np.swapaxes(
                    np.array(shape.exterior.xy), 0, 1
                ).tolist()
                polys.append(ee.Geometry.Polygon(coordinates))

        return polys


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


def get_quarters(min_water):
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


def get_bound(polygon_path, river):
    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            geom_river = feature['properties']['River']

            if geom_river != river:
                continue

            return ee.geometry.Geometry(feature['geometry'])
