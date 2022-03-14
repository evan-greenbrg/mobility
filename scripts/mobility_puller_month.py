import argparse
import os
import ee
import ee.mapclient
import fiona
import rasterio
from shapely.geometry import Polygon, LineString
from skimage import measure, draw
from shapely import ops
import pandas
import numpy as np
import geemap as geemap
import shutil
from natsort import natsorted

# ee.Authenticate()
ee.Initialize()


def getSurfaceWater(year, month, polygon):
    sw = ee.ImageCollection("JRC/GSW1_3/MonthlyHistory")
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, ]

    day = month_days[month-1]

    if month < 10:
        month = f'0{month}'
    else:
        month = str(month)

    begin = str(year) + f'-{month}' + '-01'
    end = str(year) + f'-{month}' + f'-{day}'

    return sw.filterDate(
        begin, end
    ).median().clip(
        polygon
    )


def get_baxis(p):
    mbr_points = list(zip(
        *p.minimum_rotated_rectangle.exterior.coords.xy
    ))

    mbr_lines = [
        LineString((mbr_points[i], mbr_points[i+1]))
        for i in range(len(mbr_points) - 1)
    ]

    mbr_lengths = [
        LineString((mbr_points[i], mbr_points[i+1])).length
        for i in range(len(mbr_points) - 1)
    ]
    baxis = mbr_lines[np.argmin(mbr_lengths)]
    slope = (
        (baxis.xy[0][1] - baxis.xy[0][0])
        / (baxis.xy[1][1] - baxis.xy[1][0])
    )

    return slope


def find_split(center, slope, length=2):
    xs = [center[0][0]]
    ys = [center[1][0]]
    for i in range(1, length):
        xs.append(center[0][0] + i)
        xs.append(center[0][0] - i)

        ys.append(center[1][0] + (i * slope))
        ys.append(center[1][0] - (i * slope))

    points = np.swapaxes(np.array([xs, ys]), 0, 1)
    line = LineString(points)

    return line


def split_polygon(geom, iterations):
    polys = [Polygon(geom['coordinates'][0])]
    print(iterations)
    for i in range(iterations):
        new_polys = []
        for p in polys:
            slope = get_baxis(p)
            center = p.centroid.xy
            line = find_split(center, slope, length=2)
            new_polys += [new_p for new_p in ops.split(p, line)]
        polys = new_polys
    for poly in polys:
        yield poly


def pull_esa(polygon_path, out_root, river):
    years = [i for i in range(1985, 2021)]
    months = [i for i in range(1, 13)]

    polygon_name = polygon_path.split('/')[-1].split('.')[0]

    out_paths = {}
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            geom = feature['geometry']
            out_paths['0'] = {}
            for year in years:
                print(year)
                out_paths['0'][year] = []
                for month in months:
                    poly = ee.Geometry.Polygon(geom['coordinates'])

                    river_root = out_root.format(year, month)
                    os.makedirs(river_root, exist_ok=True)

                    year_root = os.path.join(river_root, 'temps')
                    os.makedirs(year_root, exist_ok=True)
                    image = getSurfaceWater(year, month, poly)
                    out_path = os.path.join(
                        year_root,
                        f'{river}_{year}_{month}.tif'
                    )

                    geemap.ee_export_image(
                        image,
                        filename=out_path,
                        scale=30,
                        file_per_band=False
                    )
                    downloaded = os.path.exists(out_path)

                    if downloaded:
                        out_paths['0'][year].append(out_path)

                    iterations = 1
                    while not downloaded:
                        print(downloaded)
                        exists = []
                        for i, p in enumerate(split_polygon(geom, iterations)):
                            if not out_paths.get(str(i)):
                                out_paths[str(i)] = {}

                            if not out_paths[str(i)].get(year):
                                out_paths[str(i)][year] = []

                            if False in exists:
                                exists.append(False)
                                continue

                            coords = p.exterior.coords.xy
                            ee_coords = [
                                (a, b) for (a, b) in zip(coords[0], coords[1])
                            ]
                            poly = ee.Geometry.Polygon(ee_coords)
                            image = getSurfaceWater(year, month, poly)
                            os.makedirs(year_root, exist_ok=True)
                            out_path = os.path.join(
                                year_root,
                                f'{river}_{year}_{month}_{i}.tif'
                            )
                            geemap.ee_export_image(
                                image,
                                filename=out_path,
                                scale=30,
                                file_per_band=False
                            )
                            check = os.path.exists(out_path)
                            exists.append(check)
                            if check:
                                out_paths[str(i)][year].append(out_path)

                        if (
                            (len(np.unique(exists)) == 1)
                            & (np.unique(exists)[0] == True)
                        ):
                            downloaded = True
                        else:
                            if os.path.exists(year_root):
                                shutil.rmtree(year_root)
                        iterations += 1

    for key, item in out_paths.items():
        for i, paths in item.items():
            out_paths[key][i] = natsorted(paths)

    return out_paths


def clean_esa(paths):
    images = {}
    metas = {}
    for surface, years in paths.items():
        images[surface] = {}
        metas[surface] = {}
        for year, fps in years.items():
            images[surface][year] = []
            metas[surface][year] = []
            for fp in fps:
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

                images[surface][year].append(water)
                metas[surface][year].append(dsmeta)

    return images, metas


def create_mask(images):
    # Create Complete mask
    masks = {}
    for surface, years in images.items():
        for year, ims in years.items():
            for i, im in enumerate(ims):
                if i == 0:
                    masks[surface] = im
                else:
                    masks[surface] = np.add(masks[surface], im)

    return masks


def cleanChannel(masks, thresh=3000):
    clean_masks = {}
    for surface, mask in masks.items():
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
        clean_masks[surface] = labels == np.argmax(
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

    # Calculate fr
    fR = 1 - (Nb / (A * fd_b))

    # Calculate D(B+2) - D(B+1)
    DB2_DB1 = np.sum(np.abs(np.subtract(step1, step2)))

    # Calculate Zeta dt = 1
    zeta = DB2_DB1 / (2 * A * dt)

    return D, D_A, PHI, O_PHI, fR, zeta, fb


def get_mobility_monthly(images, clean_channel_belts, year_range):
    data = {
        'year': [],
        'month': [],
        'i': [],
        'D': [],
        'D/A': [],
        'Phi': [],
        'O_Phi': [],
        'fR': [],
        'zeta': [],
    }
    for i, (surface, all_years) in enumerate(images.items()):
        print()
        print(surface)

        channel_belt = clean_channel_belts[surface]

        # Find A
        A = len(np.where(channel_belt == 1)[1])

        # Need to combine all the images and create lists for years and moths.
        # Iterate over number of years
        all_images = []
        years = []
        months = []
        for year in year_range:
            ims = all_years[year]
            for month, im in enumerate(ims):
                years.append(year)
                months.append(month)
                all_images.append(im)

        for j, im in enumerate(all_images):
            # Get the baseline
            if j == 0:
                baseline = crop_raster(im.astype(int), channel_belt)

            # Get steps
            step = crop_raster(im, channel_belt)
            if j < len(all_images) - 2:
                step1 = crop_raster(all_images[j+1].astype(int), channel_belt)
                step2 = crop_raster(all_images[j+2].astype(int), channel_belt)
            else:
                step1 = np.zeros(step1.shape)
                step2 = np.zeros(step2.shape)

            if j == 0:
                fb = channel_belt - baseline

            D, D_A, PHI, O_PHI, fR, zeta, fb = get_mobility_stats(
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

        data['year'] += years
        data['month'] += months

    return pandas.DataFrame(data=data)


def main(polygon_path, out_root, keep, river, year_range):
    paths = pull_esa(polygon_path, out_root, river)
    images, metas = clean_esa(paths)
    channel_belts = create_mask(images)
    clean_channel_belts = cleanChannel(channel_belts, 100000)
    monthly_mobility = get_mobility_monthly(
        images,
        clean_channel_belts,
        year_range
    )
    monthly_mobility['dt'] = pandas.to_datetime((
        monthly_mobility['year'].astype(str)
        + '-'
        + (monthly_mobility['month'] + 1).astype(str)
    ), format='%Y-%m')

    if keep == 'true':
        keep = True
    elif keep == 'false':
        keep = False

    if not keep:
        for surface, years in paths.items():
            for year, files in years.items():
                for file in files:
                    os.remove(file)

    return monthly_mobility


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pull Mobility')
    parser.add_argument('river', metavar='r', type=str,
                        help='Name_of_the_river')

    parser.add_argument('keep',
                        choices=['true', 'false'],
                        help='Special testing value')

    parser.add_argument('poly', metavar='p', type=str,
                        help='path to the polygon file')

    parser.add_argument('out', metavar='o', type=str,
                        help='output file directory')
    parser.add_argument('start', metavar='s', type=int,
                        help='Start year')
    parser.add_argument('end', metavar='e', type=int,
                        help='End year')

    args = parser.parse_args()

    polygon_path = f'/Users/Evan/Documents/Mobility/GIS/Lhasa_area.gpkg'
    out_root = f'/Users/Evan/Documents/Mobility/GIS/surface_water'

#    yearly_mobility, monthly_mobility = main(polygon_path, out_root)

    year_range = [i for i in range(args.start, args.end + 1)]
    monthly_mobility = main(
        args.poly, 
        args.out, 
        args.keep,
        args.river,
        year_range
    )

    monthly_mobility.to_csv(
        os.path.join(args.out, f'{args.river}_monthly_mobility.csv')
    )
#
# #    month_df_0 = monthly_mobility[monthly_mobility['i'] == 0]
# #    year_df_0 = yearly_mobility[yearly_mobility['i'] == 0]
# #
# #    fig, axs = plt.subplots(2,1)
# #    axs[0].plot(year_df_0['dt'], 1 - year_df_0['fR'])
# #
# #    axs[0].plot(month_df_0['dt'], 1 - month_df_0['fR'])
# #    plt.show()
# #
# #    fig, axs = plt.subplots(2,1)
# #    axs[0].plot(year_df_0['dt'], year_df_0['O_Phi'])
# #
# #    axs[0].plot(month_df_0['dt'], month_df_0['O_Phi'])
# #    plt.show()
