import argparse
import glob
import os
import platform
from multiprocessing import set_start_method

from natsort import natsorted
import ee
from matplotlib import pyplot as plt
import numpy as np
from puller_fun import pull_watermasks
from puller_fun import pull_esa
from mobility_fun import get_mobility_rivers
from gif_fun import make_gif


ee.Initialize()

def make_gifs(rivers, root):
    for river in rivers:
        print(river)
        fp = sorted(
            glob.glob(os.path.join(root, f'{river}/*mobility.csv'))
        )[0]
        fp_in = os.path.join(
            root, f'{river}/*.tif'
        )
        fp_out = os.path.join(
            root, f'{river}/{river}_cumulative.gif'
        )
        stat_out = os.path.join(
            root, f'{river}/{river}_mobility_stats.csv'
        )
        make_gif(fp, fp_in, fp_out, stat_out)


if __name__ == '__main__':
    if platform.system() == "Darwin":
            set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Pull Mobility')
    parser.add_argument('--poly', metavar='poly', type=str,
                        help='In path for the geopackage path')

    parser.add_argument('--method', metavar='mask_method', type=str,
                        choices=['jones', 'esa'],
                        help='Do you want to calculate mobility')

    parser.add_argument('--network_method', metavar='network_method', type=str,
                        choices=['grwl', 'merit', 'largest'],
                        default='grwl',
                        help='what method do you want to use to extract the network')

    parser.add_argument('--network_path', metavar='network_path', type=str,
                        default=None,
                        help='Path to network dataset')

    parser.add_argument('--images', metavar='images', type=str,
                        choices=['true', 'false'],
                        help='Do you want to export images')

    parser.add_argument('--mobility', metavar='mobility', type=str,
                        choices=['true', 'false'],
                        help='Do you want to calculate mobility')

    parser.add_argument('--gif', metavar='gif', type=str,
                        choices=['true', 'false'],
                        help='Do you want to make the gif?')

    parser.add_argument('--out', metavar='out', type=str,
                        help='output root directory')

    args = parser.parse_args()

    export_images = False
    if args.images == 'true':
        export_images = True
    if args.method == 'jones':
        paths = pull_watermasks(
            args.poly, 
            args.out, 
            export_images, 
            args.network_method, 
            args.network_path
        )
    elif args.method == 'esa':
        paths = pull_esa(args.poly, args.out)

    if args.mobility == 'true':
        rivers = get_mobility_rivers(args.poly, paths, args.out)

    if (args.gif == 'true'):
        make_gifs(list(paths.keys()), args.out)

# 
polygoin_path = '/Users/greenberg/Documents/PHD/Writing/Mobility_Proposal/GIS/Elwha/Elwha.gpkg'
root = '/Users/greenberg/Documents/PHD/Writing/Mobility_Proposal/GIS/Elwha'
path_list = natsorted(glob.glob('/Users/greenberg/Documents/PHD/Writing/Mobility_Proposal/GIS/Elwha/Trinity/*.tif'))
river = 'elwha'
export_images = False
method = 'merit'
merit_path = '/Users/greenberg/Documents/PHD/Projects/Mobility/river_networks/channel_networks_full.shp'

# 
# 
