import argparse
import platform
from multiprocessing import set_start_method

import ee
from puller import pull_watermasks
from puller import pull_esa
from puller import get_paths
from mobility import get_mobility_rivers
from gif import make_gifs


ee.Initialize()

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

    parser.add_argument('--river', metavar='r', type=str,
                        help='River name')

    args = parser.parse_args()

    export_images = False
    if args.images == 'true':
        export_images = True
    if args.masks == 'true':
        if (args.mask_method == 'Jones') or (args.mask_method == 'Zou'):
            print('Pulling Images')
            paths = pull_watermasks(
                args.poly,
                args.out,
                args.river,
                export_images,
                args.mask_method,
                args.network_method,
                args.network_path,
                args.period
            )
        elif args.method == 'esa':
            paths = pull_esa(args.poly, args.out)
    else:
        paths = get_paths(args.poly, args.out, args.river, args.river)

    if args.mobility == 'true':
        print('Pulling Mobility')
        rivers = get_mobility_rivers(args.poly, paths, args.out, args.river)

    if (args.gif == 'true'):
        print('Making Gif')
        make_gifs(args.river, args.out)
