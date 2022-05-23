import argparse
import glob
import os
import platform
from multiprocessing import set_start_method

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
    parser.add_argument('poly', metavar='in', type=str,
                        help='In path for the geopackage path')
    parser.add_argument('method', metavar='met', type=str,
                        choices=['jones', 'esa'],
                        help='Do you want to calculate mobility')
    parser.add_argument('mobility', metavar='mob', type=str,
                        choices=['true', 'false'],
                        help='Do you want to calculate mobility')
    parser.add_argument('gif', metavar='gif', type=str,
                        choices=['true', 'false'],
                        help='Do you want to make the gif?')
    parser.add_argument('out', metavar='out', type=str,
                        help='output root directory')
    args = parser.parse_args()

    if args.method == 'jones':
        paths = pull_watermasks(args.poly, args.out)
    elif args.method == 'esa':
        paths = pull_esa(args.poly, args.out)

    if args.mobility == 'true':
        rivers = get_mobility_rivers(args.poly, paths, args.out)

    if (args.gif == 'true'):
        make_gifs(list(paths.keys()), args.out)

# 
# poly = '/Users/greenberg/Documents/PHD/Writing/Mobility_Proposal/GIS/Trinity.gpkg'
# out = '/Users/greenberg/Documents/PHD/Writing/Mobility_Proposal/GIS/TrinityUse'
# path_list = natsorted(glob.glob('/Users/greenberg/Documents/PHD/Writing/Mobility_Proposal/GIS/TrinityUse/Trinity/*.tif'))
# river = 'Trinity'


