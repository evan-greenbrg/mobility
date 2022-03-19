import copy
import os
import ee
import ee.mapclient
import fiona
import rasterio
import rasterio.mask
from skimage import measure, draw
import pandas
import numpy as np
import geemap as geemap
from natsort import natsorted


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
                downloaded = os.path.exists(out_path)
                if downloaded:
                    print()
                    print('Already Exists')
                    print()
                    out_paths[river].append(out_path)
                    continue

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
                    print(river)
                    print('Not Downloaded')
                    print()

    for key, item in out_paths.items():
        out_paths[key] = natsorted(item)

    return out_paths


def clean_esa(poly, river, fps):
    polygon_name = poly.split('/')[-1].split('.')[0]
    with fiona.open(poly, layer=polygon_name) as layer:
        for feature in layer:
            geom_river = feature['properties']['River']

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
                water = image.data[0, :, :] > 1

                dsmeta = ds.meta
                dsmeta.update(
                    width=water.shape[1],
                    height=water.shape[0],
                    count=1,
                    dtype=rasterio.int8
                )

                images[year] = water
                metas[year] = dsmeta

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

    stats = {
        'D': D,
        'D_A': D_A,
        'PHI': PHI,
        'O_PHI': O_PHI,
        'fR': fR,
        'zeta': zeta,
        'fb': fb,
        'fw_b': fw_b,
        'fd_b': fd_b,
    }

    return stats


def get_mobility_yearly(images, mask):

    A = len(np.where(mask == 1)[1])
    year_range = list(images.keys())
    ranges = [year_range[i:] for i, yr in enumerate(year_range)]
    river_dfs = {}
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
            all_images.append(images[str(year)])

        for j, im in enumerate(all_images):
            where = np.where(~mask)
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

            # Get dt
            if j < len(all_images) - 2:
                dt = int(year_range[j+2]) - int(year_range[j+1])
            else:
                dt = 1

            if j == 0:
                fb = mask - baseline

            stats = get_mobility_stats(
                j,
                A,
                mask,
                baseline,
                step,
                step1,
                step2,
                fb,
                dt
            )

            data['i'].append(j)
            data['D'].append(stats['D'])
            data['D/A'].append(stats['D_A'])
            data['Phi'].append(stats['PHI'])
            data['O_Phi'].append(stats['O_PHI'])
            data['fR'].append(stats['fR'])
            data['zeta'].append(stats['zeta'])
            data['fw_b'].append(stats['fw_b'])
            data['fd_b'].append(stats['fd_b'])
        data['year'] = years
        river_dfs[yrange[0]] = pandas.DataFrame(data=data)

    return river_dfs
