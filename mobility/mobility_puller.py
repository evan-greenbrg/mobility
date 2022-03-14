import glob
import argparse
import os
import ee
import ee.mapclient
import pandas

from puller_fun import pull_esa
from puller_fun import clean_esa
from puller_fun import create_mask_shape
from puller_fun import clean_channel_belt 
from puller_fun import filter_images
from puller_fun import get_mobility_yearly
from gif_fun import make_gif


# ee.Authenticate()
ee.Initialize()


def get_images(poly, gif, out):
    out = os.path.join(out, '{}')
    paths = pull_esa(poly, out)
    images, metas = clean_esa(paths)
    channel_belts = create_mask_shape(
        poly,
        paths
    )
    clean_channel_belts = clean_channel_belt(
        channel_belts, 100
    )
    clean_images = filter_images(
        images,
        clean_channel_belts,
        thresh=.000001
    )
    river_dfs = get_mobility_yearly(
        clean_images,
        clean_channel_belts,
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

    if gif == 'true':
        gif = True
    elif gif == 'false':
        gif = False

    if not gif:
        for river, files in paths.items():
            for f in files:
                os.remove(f)

    for river, df in full_river_dfs.items():
        df.to_csv(
            os.path.join(
                out.format(river), 
                f'{river}_yearly_mobility.csv'
            )
        )

    return list(full_river_dfs.keys())


def make_gifs(rivers, root):
    for river in rivers:
        print(river)
        fp = sorted(
            glob.glob(os.path.join(root, f'{river}/*mobility.csv'))
        )[0]
        fp_in = os.path.join(
            root, f'{river}/temps/*.tif'
        )
        fp_out = os.path.join(
            root, f'{river}/{river}_cumulative.gif'
        )
        stat_out = os.path.join(
            root, f'{river}/{river}_mobility_stats.csv'
        )
        make_gif(fp, fp_in, fp_out, stat_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pull Mobility')
    parser.add_argument('poly', metavar='in', type=str,
                        help='In path for the geopackage path')
    parser.add_argument('gif', metavar='gif', type=str,
                        choices=['true', 'false'],
                        help='Do you want to make the gif?')
    parser.add_argument('out', metavar='out', type=str,
                        help='output root directory')
    args = parser.parse_args()

    rivers = get_images(args.poly, args.gif, args.out)
    if args.gif == 'true':
        make_gifs(rivers, args.out)
