import ee




# SPECTRAL INDICES FOR SENTINEL-2 L2A

## This module facilitates index calculations in Google Earth Engine for Sentinel-2 Imagery


# 
def ndvi_ee_s2(s2_image, geom):
    '''
    Normalized Difference Vegetation Index (NDVI)

    Reference: http://dx.doi.org/10.23919/IConAC.2017.8081990

    NDVI = (NIR / RED ) / (NIR + RED)
    '''
    return s2_image.normalizedDifference(['B8', 'B4']).clip(geom).rename('NDVI')

def ndbi_ee_s2(s2_image, geom):
    '''
    Normalized Difference Built-up Index (NDBI)

    Reference: http://dx.doi.org/10.3390/rs11030345

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    '''
    return s2_image.normalizedDifference(['B11', 'B8']).clip(geom).rename('NDBI')


def bu_ee_s2(s2_image,geom):
    '''
    Built-up Index (BUI)

    Reference: http://dx.doi.org/10.3390/rs11030345

    BUI = NDBI - NDVI
    '''
    ndvi = ndvi_ee_s2(s2_image,geom)
    ndbi = ndbi_ee_s2(s2_image,geom)

    return ndbi.subtract(ndvi).clip(geom).rename('BUI')

def evi_ee_s2(s2_image, geom):
    '''
    Compute Enhanced Vegetation Index from Sentinel-2 Image
    
    Reference: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/evi/

    General formula: 
    
    EVI = 2.5 * (NIR - RED) / ((NIR + 6 * RED - 7.5 * BLUE) + 1)
    
    '''
    # Fetch Bands
  
    b8 = s2_image.select('B8').clip(geom)
    b4 = s2_image.select('B4').clip(geom)
    b2 = s2_image.select('B2').clip(geom)

    term1 = b8.subtract(b4)
    term2 = b8.add(b4.multiply(ee.Number(6)))
    term3 = b2.multiply(ee.Number(7.5))
    return term1.divide(term2.subtract(term3).add(ee.Number(1))).multiply(ee.Number(2.5)).rename('EVI')
    # return s2_image.expression(
    #     '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    #     {
    #         'NIR': s2_image.select('B8').clip(geom),
    #         'RED': s2_image.select('B4').clip(geom),
    #         'BLUE': s2_image.select('B2').clip(geom)
    #     }
    # ).rename('EVI')


def ndwi_ee_s2(s2_image, geom):
    '''
    Normalized Difference Water Index (NDWI)

    Reference: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/

    NDWI = (NIR - SWIR) / (NIR + SWIR)
    '''
    return s2_image.normalizedDifference(['B8', 'B11']).clip(geom).rename('NDWI')

