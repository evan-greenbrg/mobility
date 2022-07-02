import os
import numpy as np
import pandas

from mobility_helpers import clean_esa
from mobility_helpers import create_mask_shape


def get_mobility_rivers(poly, paths, out, river):
    print(river)
    for block, path_list in enumerate(paths):
        path_list = sorted(path_list)
    # mask stuff
        mask = create_mask_shape(
            poly,
            river,
            path_list
        )

        images, metas = clean_esa(
            poly,
            river,
            path_list
        )

        river_dfs = get_mobility_yearly(
            images,
            mask,
        )

        full_df = pandas.DataFrame()
        for year, df in river_dfs.items():
            rnge = f"{year}_{df.iloc[-1]['year']}"
            df['dt'] = pandas.to_datetime(
                df['year'],
                format='%Y'
            )
            df['range'] = rnge

            full_df = full_df.append(df)

        out_path = os.path.join(
            out,
            river,
            f'{river}_yearly_mobility_block_{block}.csv'
        )
        full_df.to_csv(out_path)

    return river


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
            'fw_b': [],
            'fd_b': [],
        }
        length = images[yrange[0]].shape[0]
        width = images[yrange[0]].shape[1]
        long = len(yrange)
        all_images = np.empty((length, width, long))
        years = []
        for j, year in enumerate(yrange):
            years.append(year)
            im = images[str(year)].astype(int)
            im[mask.mask] = 0
#            im[where] = 0
            all_images[:, :, j] = im

        baseline = all_images[:, :, 0]

        w_b = len(np.where(baseline == 1)[0])
        fb = mask - baseline
        fw_b = w_b / A
        fd_b = np.sum(fb) / A
        Na = A * fd_b

        for j in range(all_images.shape[2]):
            im = all_images[:, :, j]

            kb = (
                np.sum(all_images[:, :, :j + 1], axis=(2))
                + mask
            )
            kb[np.where(kb != 1)] = 0
            Nb = np.sum(kb)
#            fR = 1 - (Nb / (A * fd_b))
            fR = (Na / w_b) - (Nb / w_b)

            # Calculate D - EQ. (1)
            D = np.sum(np.abs(np.subtract(baseline, im)))

            # Calculate D / A
            D_A = D / A

            # Calculate Phi
            w_t = len(np.where(im == 1)[0])
            fw_t = w_t / A
            fd_t = (A - w_t) / A

            PHI = (fw_b * fd_t) + (fd_b * fw_t)

            # Calculate O_Phi
            O_PHI = 1 - (D / (A * PHI))

            data['i'].append(j)
            data['D'].append(D)
            data['D/A'].append(D_A)
            data['Phi'].append(PHI)
            data['O_Phi'].append(O_PHI)
            data['fR'].append(fR)
            data['fw_b'].append(fw_b)
            data['fd_b'].append(fd_b)

        data['year'] = years
        river_dfs[yrange[0]] = pandas.DataFrame(data=data)

    return river_dfs
