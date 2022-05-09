import numpy as np
import pandas


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
            im = images[str(year)]
            where = np.where(~mask)
            im[where] = 0
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
                np.sum(all_images[:,:, :j + 1], axis=(2)) 
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
