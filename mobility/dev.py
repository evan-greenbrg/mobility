import argparse
import glob
import os
import platform
from multiprocessing import set_start_method

from natsort import natsorted
import ee
from matplotlib import pyplot as plt
import numpy as np
from puller_fun import pull_image_watermasks
from puller_fun import pull_esa
from mobility_fun import get_mobility_rivers
from gif_fun import make_gif
from landsat_fun import getPolygon 
from channel_mask_fun import getMERITfeatures 
from puller_fun import get_months
from puller_fun import pullYearMask
from puller_fun import mosaic_images
from multiprocessing_fun import multiprocess


ee.Initialize()


if __name__ == '__main__':
    if platform.system() == "Darwin":
            set_start_method('spawn')


    polygon_path = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Development/YellowUpstream/YellowUpstream.gpkg'
    root = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Development/YellowUpstream/'
    river = 'YellowUpstream'
    export_images = False
    method = 'grwl'
    merit_path = '/Users/greenberg/Documents/PHD/Projects/Mobility/river_networks/channel_networks_full.shp'
    mask_method = 'Jones'
    network_method = 'grwl'
    period = 'quarterly'

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
#        pull_months = get_months(
#            2018, polys, year_root, river, 
#            mask_method=mask_method, 
#            network_method=network_method,
#            network=merit_path,
#            period=period
#        )
#        pull_months = [[
#            '01', '02', '03', '04', '05', '06', '07', '08',
#            '09', '10', '11', '12'
#        ]]

        # Iterate through all the years and all of the pull_months
        pull_months = [
            ['05', '06', '07'], 
            ['08', '09', '10'], 
            ['11', '12', '01'], 
            ['02', '03', '04']
        ]
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

    # open file in write mode
    with open(r'paths.txt', 'w') as fp:
        for item in out_paths:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')




poly = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Development/YellowUpstream/YellowUpstream.gpkg'
river = 'YellowUpstream'
fps = glob.glob('/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Development/YellowUpstream/YellowUpstream/mask/*')
blocks = {
    '0': [],
    '1': [],
    '2': [],
    '3': [],
}
for fp in fps:
    block = fp.split('.')[0].split('_')[-1]
    blocks[str(block)].append(fp)

all_paths = []
for block, paths in blocks.items():
    all_paths.append(paths)

paths = {}
paths['YellowUpstream'] = all_paths
