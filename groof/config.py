from pathlib import Path
import sys
sys.path.append('../../../geoml')

# Configure CORE Directories
ROOT = Path('../')
DATA = ROOT / 'data'
META = ROOT / 'meta'
RAW = DATA / 'raw'
PRC = DATA  / 'prc'

# METADATA
s2_bands_fp = 'meta/sentinel_bands.csv'

# RAW LAYERS
BBOX = RAW / 'bbox.geojson'
BLDG_FP = RAW / 'chicago_bldg.geojson'
EVI_MEDIAN = RAW / 'evi_median.tif'
BU_MEDIAN =  RAW /'bu_median.tif'
NDWI_MEDIAN = RAW / 'ndwi_median.tif'
NDVI_MEDIAN = RAW / 'ndvi_median.tif'
S2_MOSAIC = RAW / 's2.tif'
S2_MEDIAN= RAW / 's2_median.tif'
NDPI_MEDIAN = RAW / 'ndpi_median.tif'



# Reprojected Layerds
evi_reproj_fp = str(EVI_MEDIAN).replace('raw', 'prc')
bu_reproj_fp = str(BU_MEDIAN).replace('raw', 'prc')
ndwi_reproj_fp = str(NDWI_MEDIAN).replace('raw', 'prc')
ndvi_reproj_fp = str(NDVI_MEDIAN).replace('raw', 'prc')
S2_MOSAIC_REPROJ = str(S2_MOSAIC).replace('raw', 'prc')

# TRAINING DATA

GROOF_TRAIN_AREAS = RAW / 'greenroof_train.gpkg'
TRAIN = PRC / 'bldg_groof_train.tif'
TEST = PRC / 'bldg_groof_test.tif'


# S2_BANDS = [
#     'B2',   # Blue (490 nm)
#     'B3',   # Green (560 nm)
#     'B4',   # Red (665 nm)
#     'B5',   # Vegetation Red Edge (705 nm)
#     'B6',   # Vegetation Red Edge (740 nm)
#     'B7',   # Vegetation Red Edge (783 nm)
#     'B8',   # NIR (842 nm)
#     'B8A',  # Narrow NIR (865 nm)
#     'B9',   # Water vapor (945 nm)
#     'B11',  # SWIR (1610 nm)
#     'B12',  # SWIR (2190 nm)))
#     ]

S2_BANDS =  [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
    'AOT',  # Aerosol Optical Thickness
    'WVP',  # Water Vapor Pressure
    'SCL',  # Scene Classification Layer
    'QA60', # Cloud Mask (bitmask)
]