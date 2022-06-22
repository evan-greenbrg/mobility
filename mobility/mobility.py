import argparse
import glob
import os
import platform
from multiprocessing import set_start_method
import warnings

from natsort import natsorted
import ee
from matplotlib import pyplot as plt
import numpy as np
from puller_fun import pull_image_watermasks
from puller_fun import pull_esa
from puller_fun import get_paths 
from mobility_fun import get_mobility_rivers
from gif_fun import make_gif


ee.Initialize()
warnings.filterwarnings("ignore")

def make_gifs(rivers, root):
    for river in rivers:
        print(river)
        fps = sorted(
            glob.glob(os.path.join(root, f'{river}/*mobility_block*.csv'))
        )
        fp_in = os.path.join(
            root, f'{river}/mask/*block_0.tif'
        )
        fp_out = os.path.join(
            root, f'{river}/{river}_cumulative.gif'
        )
        stat_out = os.path.join(
            root, f'{river}/{river}_mobility_stats.csv'
        )
        make_gif(fps, fp_in, fp_out, stat_out)


if __name__ == '__main__':
    if platform.system() == "Darwin":
            set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Pull Mobility')
    parser.add_argument('--poly', metavar='poly', type=str,
                        help='In path for the geopackage path')

    parser.add_argument('--mask_method', metavar='mask_method', type=str,
                        choices=['Jones', 'esa', 'Zou'],
                        help='Do you want to calculate mobility')

    parser.add_argument('--network_method', metavar='network_method', type=str,
                        choices=['grwl', 'merit', 'largest', 'all'],
                        default='grwl',
                        help='what method do you want to use to extract the network')

    parser.add_argument('--network_path', metavar='network_path', type=str,
                        default=None,
                        help='Path to network dataset')

    parser.add_argument('--masks', metavar='images', type=str,
                        choices=['true', 'false'],
                        help='Do you want to export masks')

    parser.add_argument('--images', metavar='images', type=str,
                        choices=['true', 'false'],
                        help='Do you want to export images')

    parser.add_argument('--mobility', metavar='mobility', type=str,
                        choices=['true', 'false'],
                        help='Do you want to calculate mobility')

    parser.add_argument('--gif', metavar='gif', type=str,
                        choices=['true', 'false'],
                        help='Do you want to make the gif?')

    parser.add_argument('--period', metavar='images', type=str,
                        choices=[
                            'annual', 
                            'quarterly', 
                            'bankfull', 
                            'max', 
                            'min'
                        ],
                        help='Do you want to export images')

    parser.add_argument('--out', metavar='out', type=str,
                        help='output root directory')

    args = parser.parse_args()

    export_images = False
    if args.images == 'true':
        export_images = True
    if args.masks == 'true':
        if (args.mask_method == 'Jones') or (args.mask_method == 'Zou'):
            print('Pulling Images')
            paths = pull_image_watermasks(
                args.poly, 
                args.out, 
                export_images, 
                args.mask_method, 
                args.network_method, 
                args.network_path,
                args.period
            )
        elif args.method == 'esa':
            paths = pull_esa(args.poly, args.out)
    else:
        paths = get_paths(args.poly, args.out)

    if args.mobility == 'true':
        print('Pulling Mobility')
        rivers = get_mobility_rivers(args.poly, paths, args.out)

    if (args.gif == 'true'):
        print('Making Gif')
        make_gifs(list(paths.keys()), args.out)



# root ="/Users/greenberg/Documents/PHD/Projects/Mobility/Parameter_space/PNG/"
# rivers = [
#     'PNG2',
#     'PNG3',
#     'PNG4d',
#     'PNG4u',
#     'PNG5',
#     'PNG6',
#     'PNG7',
#     'PNG9',
#     'PNG10',
# ]
# make_gifs(rivers, out)

