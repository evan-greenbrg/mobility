import argparse
import os
import ee
import ee.mapclient
import fiona
import rasterio
import rasterio.mask
from shapely.geometry import Polygon, LineString
from skimage import measure, draw
from shapely import ops
import pandas
import numpy as np
import geemap as geemap
import shutil
from natsort import natsorted
from matplotlib import pyplot as plt

# ee.Authenticate()
ee.Initialize()


def getSurfaceWater(year, polygon):
    sw = ee.ImageCollection("JRC/GSW1_3/YearlyHistory")

    begin = str(year) + f'-01' + '-01'
    end = str(year) + f'-12' + f'-31'

    return sw.filterDate(
        begin, end
    ).median().clip(
        polygon
    )


def pull_esa(polygon_path, out_root):
    years = [i for i in range(1985, 2020)]

    polygon_name = polygon_path.split('/')[-1].split('.')[0]

    out_paths = {}
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            geom = feature['geometry']
            river = feature['properties']['River']
            out_paths[river] = []
            for year in years:
                print(year)
                poly = ee.Geometry.Polygon(geom['coordinates'])

                river_root = out_root.format(river)
                os.makedirs(river_root, exist_ok=True)

                year_root = os.path.join(river_root, 'temps')
                os.makedirs(year_root, exist_ok=True)
                image = getSurfaceWater(year, poly)
                out_path = os.path.join(
                    year_root,
                    f'{river}_{year}.tif'
                )

                geemap.ee_export_image(
                    image,
                    filename=out_path,
                    scale=30,
                    file_per_band=False
                )
                downloaded = os.path.exists(out_path)

                if downloaded:
                    out_paths[river].append(out_path)
                else:
                    print()
                    print()
                    print('Not Downloaded')
                    print()
                    print()


    for key, item in out_paths.items():
        out_paths[key] = natsorted(item)

    return out_paths


def clean_esa(paths):
    images = {}
    metas = {}
    for river, fps in paths.items():
        images[river] = {}
        metas[river] = {}
        for fp in fps:
            year = fp.split('_')[-1].split('.')[0]
            ds = rasterio.open(fp)
            image = ds.read(1)

            water = image > 1

            dsmeta = ds.meta
            dsmeta.update(
                width=water.shape[1],
                height=water.shape[0],
                count=1,
                dtype=rasterio.int8
            )

            images[river][year] = water
            metas[river][year] = dsmeta

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


def create_mask_shape(polygon_path, paths):
    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        masks = {}
        for feature in layer:
            river = feature['properties']['River']
            geom = feature['geometry']

            image = paths[river][0]
            ds = rasterio.open(image)
            out_image, out_transform = rasterio.mask.mask(
                ds, [geom], 
                crop=False, filled=False
            )
            out_image += 11
            out_image[np.where(out_image < 10)] = 0
            out_image[np.where(out_image > 10)] = 1

            masks[river] = out_image[0,:,:]

    return masks



def cleanChannel(masks, thresh=3000):
    clean_masks = {}
    for river, mask in masks.items():
        print(river)
        labels = measure.label(mask)
        # assume at least 1 CC
        assert(labels.max() != 0)
        # Find largest connected component
        channel = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        channel = fillHoles(channel, thresh)

        labels = measure.label(channel)
        # assume at least 1 CC
        assert(labels.max() != 0)
        # Find largest connected component
        clean_masks[river] = labels == np.argmax(
            np.bincount(labels.flat)[1:]
        ) + 1

    return clean_masks


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


def get_mobility_stats(j, A, channel_belt, baseline,
                       step, step1, step2, fb, dt=1):
    # Calculate D - EQ. (1)
    D = np.sum(np.abs(np.subtract(baseline, step)))

    # Calculate D / A
    D_A = D / A

    # Calculate Phi
    w_b = len(np.where(baseline == 1)[0])
    w_t = len(np.where(step == 1)[0])

    fw_b = w_b / A
    fw_t = w_t / A
    fd_b = (A - w_b) / A
    fd_t = (A - w_t) / A

    PHI = (fw_b * fd_t) + (fd_b * fw_t)

    # Calculate O_Phi
    O_PHI = 1 - (D / (A * PHI))

    # Calculate Np
    fb = fb - step
    Nb = len(np.where(fb == 1)[0])

    # THIS CALCULATION IS INCORRECT
    # Calculate fr
    fR = 1 - (Nb / (A * fd_b))

    # Calculate D(B+2) - D(B+1)
    DB2_DB1 = np.sum(np.abs(np.subtract(step1, step2)))

    # Calculate Zeta dt = 1
    zeta = DB2_DB1 / (2 * A * dt)

    return D, D_A, PHI, O_PHI, fR, zeta, fb, fw_b, fd_b


def get_mobility_yearly(images, clean_channel_belts, year_range):
    river_dfs = {}
    for i, (river, all_years) in enumerate(images.items()):
        river_dfs[river] = {}
        if not len(all_years):
            continue
        print()
        print(river)

        channel_belt = clean_channel_belts[river]

        # Find A
        A = len(np.where(channel_belt == 1)[1])

        # Combine the year steps
        ranges = [year_range[i:] for i, yr in enumerate(year_range)]
        for yrange in ranges:
            data = {
                'year': [],
                'i': [],
                'D': [],
                'D/A': [],
                'Phi': [],
                'O_Phi': [],
                'fR': [],
                'zeta': [],
                'fw_b': [],
                'fd_b': [],
            }
            all_images = []
            years = []
            for year in yrange:
                years.append(year)
                all_images.append(all_years[str(year)])

            for j, im in enumerate(all_images):
                where = np.where(~channel_belt)
                im[where] = 0
                # Get the baseline
                if j == 0:
                    baseline = im.astype(int)

                # Get step
                step = im.astype(int)
                if j < len(all_images) - 2:
                    step1 = all_images[j+1].astype(int)
                    step2 = all_images[j+2].astype(int)
                else:
                    step1 = np.zeros(step1.shape)
                    step2 = np.zeros(step2.shape)

                if j == 0:
                    fb = channel_belt - baseline

                D, D_A, PHI, O_PHI, fR, zeta, fb, fw_b, fd_b = get_mobility_stats(
                    j,
                    A,
                    channel_belt,
                    baseline,
                    step,
                    step1,
                    step2,
                    fb
                )

                data['i'].append(i)
                data['D'].append(D)
                data['D/A'].append(D_A)
                data['Phi'].append(PHI)
                data['O_Phi'].append(O_PHI)
                data['fR'].append(fR)
                data['zeta'].append(zeta)
                data['fw_b'].append(fw_b)
                data['fd_b'].append(fd_b)
            data['year'] = years
            river_dfs[river][yrange[0]] = pandas.DataFrame(data=data)

    return river_dfs 


def main(polygon_path, out_root, keep, year_range):
    paths = pull_esa(polygon_path, out_root)
    images, metas = clean_esa(paths)
    # Implemented method
#    channel_belts = create_mask(images)
    # New method
    channel_belts = create_mask_shape(polygon_path, paths)
    clean_channel_belts = cleanChannel(channel_belts, 100000)
    river_dfs = get_mobility_yearly(
        images,
        clean_channel_belts,
        year_range
    )

    full_river_dfs = {}
    for river, years in river_dfs.items():
        full_df = pandas.DataFrame()
        for year, df in years.items():
            rnge = f"{year}_{df.iloc[-1]['year']}"
            df['dt'] = pandas.to_datetime(
                df['year'],
                format='%Y'
            )
            df['range'] = rnge

            full_df = full_df.append(df)

        full_river_dfs[river] = full_df

    if keep == 'true':
        keep = True
    elif keep == 'false':
        keep = False

    if not keep:
        for river, files in paths.items():
            for f in files:
                os.remove(f)

    return full_river_dfs 


if __name__ == '__main__':
    polygon_path = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Development/MigrationRateCompare.gpkg'
    out_root = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Comparing/big_compare/box_based_channel_belt/{}'
    year_range = [i for i in range(1990, 2020)]
    keep = 'true'

    river_dfs = main(
        polygon_path, 
        out_root, 
        keep,
        year_range
    )

    for river, df in river_dfs.items():
        df.to_csv(
            os.path.join(
                out_root.format(river), 
                f'{river}_yearly_mobility.csv'
            )
        )
