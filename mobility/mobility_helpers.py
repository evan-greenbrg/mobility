import numpy as np
import fiona
import rasterio


def create_mask_shape(polygon_path, river, fps):
    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            geom = feature['geometry']

            image = fps[0]
            ds = rasterio.open(image)
            out_image, out_transform = rasterio.mask.mask(
                ds, [geom],
                crop=False, filled=False
            )
            out_image = out_image.astype('int64')
            out_image += 11
            out_image[np.where(out_image < 10)] = 0
            out_image[np.where(out_image > 10)] = 1

            return out_image[0, :, :]


def clean_esa(poly, river, fps):
    polygon_name = poly.split('/')[-1].split('.')[0]
    with fiona.open(poly, layer=polygon_name) as layer:
        for feature in layer:
            geom = feature['geometry']

    images = {}
    metas = {}
    for fp in fps:
        year = fp.split('/')[-1].split('_')[1]
        ds = rasterio.open(fp)

        image, tf = rasterio.mask.mask(
            ds, [geom],
            crop=False, filled=False
        )

        # Threshold
        water = image.data[0, :, :] > 0
        if not np.sum(water):
            continue

        meta = ds.meta
        meta.update(
            width=water.shape[1],
            height=water.shape[0],
            count=1,
            dtype=rasterio.int8
        )

        images[year] = water
        metas[year] = meta

        with rasterio.open(fp, "w", **meta) as dest:
            dest.write(water.astype(rasterio.int8), 1)

    return images, metas
