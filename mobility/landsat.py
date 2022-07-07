import os
import ee
import ee.mapclient

# ee.Authenticate()
ee.Initialize()


def maskL8sr(image):
    """
    Masks out clouds within the images
    """
    # Bits 3 and 5 are cloud shadow and cloud
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get pixel QA band
    qa = image.select('BQA')
    # Both flags should be zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
        qa.bitwiseAnd(cloudsBitMask).eq(0)
    )

    return image.updateMask(mask)


def getLandsatCollection():
    """
    merge landsat 5, 7, 8 collection 1
    tier 1 SR imageCollections and standardize band names
    """
    # standardize band names
    bn8 = ['B1', 'B2', 'B3', 'B4', 'B6', 'pixel_qa', 'B5', 'B7']
    bn7 = ['B1', 'B1', 'B2', 'B3', 'B5', 'pixel_qa', 'B4', 'B7']
    bn5 = ['B1', 'B1', 'B2', 'B3', 'B5', 'pixel_qa', 'B4', 'B7']
    bns = ['uBlue', 'Blue', 'Green', 'Red', 'Swir1', 'BQA', 'Nir', 'Swir2']

    # create a merged collection from landsat 5, 7, and 8
    ls5 = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR").select(bn5, bns)

    ls7 = (ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")
           .filterDate('1999-04-15', '2003-05-30')
           .select(bn7, bns))

    ls8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").select(bn8, bns)

    merged = ls5.merge(ls7).merge(ls8)

    return(merged)


def get_image(year, polygon):
    """
    Set up server-side image object
    """
    # Get begining and end
    begin = str(year) + '-01' + '-01'
    end = str(year) + '-12' + '-31'

    band_names = ['uBlue', 'Blue', 'Green', 'Red', 'Swir1', 'Nir', 'Swir2']
    allLandsat = getLandsatCollection()

    # Filter image collection by
    return allLandsat.map(
        maskL8sr
    ).filterDate(
        begin, end
    ).median().clip(
        polygon
    ).select(band_names)


def get_month_image_all_polys(year, polys, month):
    months = {
        '01': '31',
        '02': '28',
        '03': '31',
        '04': '30',
        '05': '31',
        '06': '30',
        '07': '31',
        '08': '31',
        '09': '30',
        '10': '31',
        '11': '30',
        '12': '31',
    }
    day = months[month]
    begin = str(year) + '-' + month + '-01'
    end = str(year) + '-' + month + '-' + day

    band_names = [
        'uBlue', 'Blue', 'Green', 'Red', 'Swir1', 'Nir', 'Swir2'
    ]
    allLandsat = getLandsatCollection()
    for poly in polys:
        yield allLandsat.map(
            maskL8sr
        ).filterDate(
            begin, end
        ).median().clip(
           poly
        ).select(band_names)


def get_image_specific_months(year, pull_months, polygon):
    """
    Set up server-side image object
    """
    # Get begining and end
    months = {
        '01': '31',
        '02': '28',
        '03': '31',
        '04': '30',
        '05': '31',
        '06': '30',
        '07': '31',
        '08': '31',
        '09': '30',
        '10': '31',
        '11': '30',
        '12': '31',
    }
    images = ee.List([])
    for month in pull_months:
        begin = str(year) + '-' + month + '-01'
        end = str(year) + '-' + month + '-' + months[month]

        allLandsat = getLandsatCollection()

        # Filter image collection by
        images = images.add(allLandsat.map(
            maskL8sr
        ).filterDate(
            begin, end
        ).median().clip(
            polygon
        ))

    return ee.ImageCollection(
        images
    ).median().clip(
        polygon
    )


def request_params(filename, scale, image):
    filename = os.path.abspath(filename)
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]

    params = {"name": name, "filePerBand": False}
    params["scale"] = scale
    region = image.geometry()
    params["region"] = region

    return params


def surface_water_image(year, polygon):
    sw = ee.ImageCollection("JRC/GSW1_3/YearlyHistory")

    begin = str(year) + '-01' + '-01'
    end = str(year) + '-12' + '-31'

    return sw.filterDate(
        begin, end
    ).median().clip(
        polygon
    )
