
import ee
import numpy as np
from prettytable import PrettyTable
import os
import geemap



def within_bounds(asset, geom):
    
    # check if asset in an ee.Image
    if isinstance(asset , ee.Image):
        asset = ee.ImageCollection(asset)

    asset=asset.filterBounds(geom).limit(10)
    not_empty = True if asset.size().gt(0).getInfo() == 1 else False

    return not_empty

ds_map = dict(
    S2 = 'Sentinel-2 L2A [S2]',
    IS = 'Impervious Surface Fraction [IS]',
    BH = 'Building Height [BH]',
    CH = 'Tree Canopy Height [CH]',
    DM = 'Digital Surface Model (DM)'
)


def minmax_rescale(image,band, geom):
    
    # Compute min and max using reduceRegion
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=geom,
        scale=image.projection().nominalScale(),          
        maxPixels=1e9
    )

    # Get the actual min and max values
    min_val = stats.getNumber(f'{band}_min')
    max_val = stats.getNumber(f'{band}_min')

    # Print min/max for debugging
    print('Min:', min_val.getInfo())
    print('Max:', max_val.getInfo())

    # Apply dynamic min-max scaling: (img - min) / (max - min)
    return image.subtract(min_val).divide(max_val.subtract(min_val))

## SENTINEL-2 FUNCTIONS
## ================================================================================

def s2_processing(col, cloud_threshold):
    return col.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',cloud_threshold))

def ee_get_s2(ee_s2, geom, start_date, end_date,max_cloud):

    bands = ee_s2['bands']
    s2_col = ee_read(ee_s2['id']).filterBounds(geom).filterDate(start_date,end_date)
    s2_col = s2_processing(s2_col, max_cloud)

    return s2_col.map(lambda img: img.select(bands))

def s2_nearest_date(asset, date):
    '''
    Fetches an image at the nearest date
    '''
    if isinstance(asset, ee.Image):
        asset=ee.ImageCollection(asset)

    date_ms= date.millis().getInfo()
    col_dates=asset.aggregate_array('system:time_start').getInfo()

    if len(col_dates) > 1:
        time_delta= [abs(ts - date_ms) for ts in col_dates]
        min_delta = min(time_delta)
        matching_date = ee.Date(col_dates[time_delta.index(min_delta)])
    else:
        matching_date = ee.Date(col_dates[0])
   

    asset = asset.filterDate(matching_date, matching_date.advance(1, 'month'))
    start_date, end_date = get_date_range(asset)
    
    return asset.median(), start_date, end_date

## UT-GLOBUS FUNCTIONS
## ================================================================================

def fetch_utglobus_id(city):
    
    city = "_".join(city.lower().split(" "))
    assets= ee.data.listAssets("projects/sat-io/open-datasets/UT-GLOBUS/")['assets']

    ids=[asset['id'] for asset in assets]
    cities = [a_id.split("/")[-1] for a_id in ids]

    
    if city in cities:
        city_idx=cities.index(city)
        return ids[city_idx]
    else:
        print(f'- {city} Not Found in UT-GLOBUS Dataset ')
        return None


def ee_rasterize(asset, val):
    return( asset
    .filter(ee.Filter.notNull([val]))
    .reduceToImage(
        reducer= ee.Reducer.first(),
        properties=[val],
        )
        .unmask(0))

# def utglobus_processor(asset, resolution):

#     raster= ee_rasterize(asset)
#     # Convert building vectors to raster
#     return (raster
#         .setDefaultProjection(
#         crs= 'EPSG:4326',
#         scale= 10
#         )
#         .reduceResolution(
#         reducer= ee.Reducer.mean()
#         ).reproject(
#             crs='EPSG:4326', 
#             scale = resolution
#         )).rename('BH')


## URBAN CANOPY PARAMETER FUNCTIONS
## ================================================================================

GISA_DICT = {
    1972: 1, 1978: 2, 1985: 3, 1986: 4, 1987: 5, 1988: 6, 1989: 7, 1990: 8, 1991: 9, 1992: 10,
    1993: 11, 1994: 12, 1995: 13, 1996: 14, 1997: 15, 1998: 16, 1999: 17, 2000: 18, 2001: 19,
    2002: 20, 2003: 21, 2004: 22, 2005: 23, 2006: 24, 2007: 25, 2008: 26, 2009: 27, 2010: 28,
    2011: 29, 2012: 30, 2013: 31, 2014: 32, 2015: 33, 2016: 34, 2017: 35, 2018: 36, 2019: 37
}


def esa_wc_compute_isf(image,resolution):
    target_val = 50 # Value for built-up class is 50
    imp = image.eq(target_val) # Get Impervious Surface Area
    return average_resampling(imp, resolution) # Output Impervious Surface Fraction


def gisa_processor(image, year, resolution):
    is_val = GISA_DICT[year]
    masked = image.updateMask(image.lte(is_val)).unmask(0)
    return average_resampling(masked,resolution)

def basic(asset, resolution):
    return asset

def building_surface_fraction(image, resolution):
    binary = image.gt(0).reproject('EPSG:4326', scale=10) # Get binary mask of buildings 
    return binary.reduceResolution(reducer= ee.Reducer.mean()).reproject(crs= 'EPSG:4326' , scale=resolution).rename("BS") ## Average Resampling to 30m


def average_resampling(image,resolution):
    return image.reproject('EPSG:4326', scale=10).reduceResolution(reducer= ee.Reducer.mean()).reproject(crs= 'EPSG:4326' , scale=resolution) ## Average Resampling to 30m


def select_ucp_source(ucp_info, geom, target_year):
    # Check which assets are within bounds
    asset_ids = [asset['id'] for asset in ucp_info]
    assets= [ee_read(asset_id) for asset_id in asset_ids]
    is_within = [within_bounds(asset,geom) for asset in assets]

    # Check closest items within years
    years = [asset['years'] for asset in ucp_info]
    year_deltas = [abs(target_year - np.array(year)) for year in years]
    min_deltas = [min(year_delta) for year_delta in year_deltas]
    min_delta_filtered= [min_delta for min_delta, within in zip(min_deltas, is_within) if within]
    min_delta = min(min_delta_filtered)
    
    sel = list((np.array(is_within)) & (np.array(min_deltas) == np.int64(min_delta)))

    sel_idx = sel.index(True)
    sel_deltas = year_deltas[sel_idx]
    nearest_year = years[sel_idx][list(sel_deltas).index(min_delta)]

    print(f'SELECTED UCP SOURCE : {ucp_info[sel_idx]["name"]}')
    print(f'SELECTED YEAR : {nearest_year}')

    sel_ucp_info = ucp_info[sel_idx]
    sel_ucp_info.update(dict(nearest_year = nearest_year))
    return sel_ucp_info




    
def read_ucp(ucp_info, geom , resolution, rescale = False):
    '''
     Read Datasets from Google Earth Engine only if within target bounds
    '''
    ucp_id, name, asset_id, band, processor, years, nearest_year  = [ucp_info[key] for key in ucp_info.keys()]

    
    print(f'=============== {name} ================')
    asset=ee_read(asset_id)
    print('- Read Successful')

    ## Handling for ee.Image
    if isinstance(asset, ee.Image): 
        image =  asset.select(band).clip(geom) ## Add clipped image
        print('- Filtered to Region of Interest')
        if 'GISA_1972_2021' in asset_id:
            processed_image = processor(image,nearest_year,resolution)
        else:
            processed_image= processor(image, resolution)
        print(f'- Processor applied: {processor}') 
    else:
        date = ee.Date(str(nearest_year))

        if 'UT-GLOBUS' in asset_id: # check if UT-GLOBUS Dataset
            asset = asset.filterBounds(geom)
            raster = ee_rasterize(asset, 'height')
            processed_image = processor(raster, resolution) 
            print('- Filtered to Region of Interest') 
            print(f'- Processor applied: {processor}') 
        
        else:
            image = asset.filterBounds(geom).filterDate(date, date.advance(1, 'year')).select(band).mosaic()
            print('- Filtered to Region of Interest') 
            
            processed_image = processor(image, resolution)
            print(f'- Processor applied: {processor}') 

    processed_image= processed_image.unmask(0).clip(geom).rename(ucp_id)

    if rescale:
        processed_image =minmax_rescale(processed_image,ucp_id,geom)

    ucp_info_out = ucp_info.copy()
    ucp_info_out.update(dict(
        image = image,
        ucp = processed_image
    ))
     
    print(f'- Image added') 
    
    return ucp_info_out

def ee_fetch_roads(geometry):
    dataset_paths = {
        'Africa': "projects/sat-io/open-datasets/GRIP4/Africa",
        'Central-South-America': "projects/sat-io/open-datasets/GRIP4/Central-South-America",
        'Europe': "projects/sat-io/open-datasets/GRIP4/Europe",
        'North-America': "projects/sat-io/open-datasets/GRIP4/North-America",
        'Oceania': "projects/sat-io/open-datasets/GRIP4/Oceania",
        'South-East-Asia': "projects/sat-io/open-datasets/GRIP4/South-East-Asia",
        'Middle-East-Central-Asia': "projects/sat-io/open-datasets/GRIP4/Middle-East-Central-Asia"
    }

    filtered_collections = []

    for region, path in dataset_paths.items():
        fc = ee.FeatureCollection(path).filterBounds(geometry)
        # Only include the FC if it has features
        count = fc.size()
        if count.getInfo() > 0:
            print(f"Including GRIP4: {region} ({count.getInfo()} features)")
            filtered_collections.append(fc)

    # Merge all non-empty collections
    if filtered_collections:
        merged = filtered_collections[0]
        for fc in filtered_collections[1:]:
            merged = merged.merge(fc)
        return merged
    else:
        print("No GRIP4 data found within the specified geometry.")
        return ee.FeatureCollection([])  # return empty FC


## DATASET DICTIONARIES
## ================================================================================

    
# MULTISPECTRAL IMAGERY
ee_s2=dict(
            name = 'Sentinel-2 L2A Harmonized',
            id = 'COPERNICUS/S2_SR_HARMONIZED',
            bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
            type='ImageCollection',
         )

# URBAN CANOPY PARAMETERS
ee_ucps = dict(
    # Impervious Surface Fraction
    IS = [
        dict(
            ucp_id = 'IS',
            name = 'ESA WorldCover 10m v200',
            id ='ESA/WorldCover/v200',
            band = 'Map',
            processor= esa_wc_compute_isf,
            years  = [2021]
        ),
       
    ],

    # Building Heights
    BH = [
        dict(
            ucp_id = 'BH',
            name = 'Google Open Research Building Heights 2.5D',
            id = 'GOOGLE/Research/open-buildings-temporal/v1',
            band = 'building_height',
            processor= average_resampling,
            years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        ),
        dict(
            ucp_id = 'BH',
            name = 'GLObal Building heights for Urban Studies (UT-GLOBUS)',
            id = None, # Updated when city parameter is intialized
            band = 'height',
            processor= average_resampling,
            years = [2020]
        ),
        dict(
            ucp_id = 'BH',
            name = 'GHSL: Global building height 2018 (P2023A)',
            id = 'JRC/GHSL/P2023A/GHS_BUILT_H/2018',
            band = 'built_height',
            processor= average_resampling,
            years = [2018]
        ),
    ],
    
    # Digital Surface Model - for Sky View Factors
    DM = [
        dict(
            ucp_id = 'DM',
            name = 'JAXA ALOS Digital Surface Model (DSM)',
            id = 'JAXA/ALOS/AW3D30/V3_2',
            band = 'DSM',
            processor= average_resampling,
            years = [2006]
        )
    ],

    # Tree Canopy Height
    CH = [
            dict(
            ucp_id = 'CH',
            name = 'Tree Canopy Height',
            id = 'users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1',
            band = 'b1',
            processor= average_resampling,
            years = [2020]
            
        )
    ]
    
)


def check_ids(ucp_dict):
    ucp_keys = ucp_dict.keys()
    clean_dict = dict()
    for key in ucp_keys:
        ucp_info = ee_ucps[key]
        clean_dict.update({key : [ucp for ucp in ucp_info if ucp['id'] is not None]})

    return clean_dict


ee_readers=dict(
    IMAGE_COLLECTION = ee.ImageCollection,
    IMAGE= ee.Image,
    TABLE = ee.FeatureCollection
)

def ee_read(id):
    '''
    Read a Google Earth Engine by only providing the asset ID
    '''

    type = ee.data.getInfo(id)['type']
    reader=ee_readers[type]

    return reader(id)

def within_date_range(col, date):

    date = date.millis().getInfo()
    start = col.first().get('system:time_start').getInfo()
    end = col.sort('system:time_start', False).first().date().millis().getInfo()

    return date >= start and date <= end


## Get Bounds  of Study Area
def get_city_bounds(city):
    admin2 = ee.FeatureCollection(GAUL_ID) # Load GAUL Level 2 dataset
    geom = admin2.filter(ee.Filter.eq('ADM1_NAME', city)) # Filter for administrative area
    
    if geom.size().getInfo() > 0:
        return geom

    else:
        return admin2.filter(ee.Filter.eq('ADM2_NAME', city)) # Filter for administrative area

def get_date_range(asset):

    if isinstance(asset , ee.Image):
        asset = ee.ImageCollection(asset)

    start = ee.Date(asset.first().get('system:time_start').getInfo())
    end = ee.Date(asset.sort('system:time_start', False).first().date().millis().getInfo())

    return start,end


def get_image(asset,band, date):
    '''
    Fetches an image at the nearest date
    '''    
    date= ee.Date(str(date))
    return asset.filterDate(date, date.advance(1,'year')).select(band).mean()



# def ee_get_image(col_id, band,bbox):
    
#     geom=ee.Geometry.Rectangle(bbox)

#     asset_type=ee.data.getInfo(col_id)['type']

#     if asset_type == 'IMAGE_COLLECTION':
#         image=ee.ImageCollection(col_id).filterBounds(geom).select(band).mean().clip(geom)
#     else:
#         image=ee.Image(col_id).select(band).clip(geom)

#     image_id = col_id.replace('/','_') + f'_{band}'
   
#     return image, image_id


def summary(images):
    # Specify the Column Names while initializing the Table
    myTable = PrettyTable(["Name", "Band", "Selected Year", "End Date", 'Time Delta' ])

    # Add rows
    for data in images:
        myTable.add_row([data['name'], data['band'], data['nearest_year']])


    print(myTable)


def lcz_gdf_to_raster(lcz_gdf, geom, legend):
    class_dict = legend.set_index('class_id')['class'].to_dict()
    geo = lcz_gdf.__geo_interface__['features']
    polys = [ee.Geometry.Polygon([g[:-1] for g in js['geometry']['coordinates'][0]]) for js in geo]
    # ee.Geometry(json.dumps(geo[0]['geometry']))
    props = [{'Name' : class_dict[(js['properties']['Name'])]} for js in geo]
    features = [ee.Feature(poly, prop) for poly, prop in zip(polys, props)]
    lcz_fc = ee.FeatureCollection(features)
    return ee_rasterize(lcz_fc, 'Name').clip(geom).rename('LCZ')



def random_forest(X, y, n_trees, class_name, region, n_samples):
    classPoints = int(n_samples / 2)

    # Combine features and labels
    input_features = X.addBands(y.toByte())

    # Stratified sampling (balanced classes)
    sample = input_features.stratifiedSample(
        numPoints=n_samples,
        classBand=class_name,
        classValues=[0, 1],
        classPoints=[classPoints, classPoints],
        region=region,
        scale=10,
        seed=42,
        geometries=False
    )

    # Add random column for splitting
    sample = sample.randomColumn()

    # 80/20 train/validation split
    trainingSample = sample.filter(ee.Filter.lte('random', 0.8))
    validationSample = sample.filter(ee.Filter.gt('random', 0.8))

    # Train Random Forest
    classifier = ee.Classifier.smileRandomForest(n_trees).train(
        features=trainingSample,
        classProperty=class_name,
        inputProperties=X.bandNames()
    )

    # Print model explanation
    print("Trained classifier explanation:", classifier.explain().getInfo())

    # Training accuracy
    train_accuracy = classifier.confusionMatrix()
    print("Training error matrix:", train_accuracy.getInfo())
    print("Training overall accuracy:", train_accuracy.accuracy().getInfo())

    # Validation accuracy
    classified_validation = validationSample.classify(classifier)
    validation_accuracy = classified_validation.errorMatrix(class_name, 'classification')
    print("Validation error matrix:", validation_accuracy.getInfo())
    print("Validation overall accuracy:", validation_accuracy.accuracy().getInfo())

    return classifier


def predict_impervious(s2_image,roads,buildings,canopy_height, geometry):


    print('Predicting Impervious Surfaces from Buildings, Roads and Canopy Height Data')
    plants = canopy_height.gt(0)


    # Add NDBI and canopy mask
    ndbi = s2_image.normalizedDifference(['B6', 'B5']).rename('NDBI')
    X = s2_image.addBands(ndbi).addBands(plants)

    # Rasterize roads
    roads_raster = roads.reduceToImage(
        reducer = ee.Reducer.max(),
        properties =  ['GP_RAV']
    ).reproject(crs='EPSG:4326', scale=5).gt(0).unmask(0).clip(geometry)

    # Rasterize buildings
    buildings_mask = buildings.gt(0).unmask(0).clip(geometry)

    # Impervious surface = roads OR buildings
    impervious = roads_raster.Or(buildings_mask).rename('impervious')

    classifier = random_forest(X, impervious, 100, 'impervious', geometry, 5000)
    # Classify the image
    classified = X.classify(classifier)

    # Apply smoothing
    classified_smooth = classified.focalMode(radius=1, units='pixels', kernelType='square')
    return classified_smooth



def ee_download_tiled_image(image, image_id,tiles_gdf,crs, scale, output_dir,):

    for idx, tile in tiles_gdf.iterrows():
        tile_id=tile.tile_id
        bbox=tile.geometry.bounds
        bbox_geom=ee.Geometry.Rectangle(bbox)
        filename=f"{output_dir}/{image_id}_{tile_id}_{scale}m.tif"
        print(filename)
        if os.path.exists(filename) == False:

            geemap.ee_export_image(
                image,
                filename=filename,
                scale=scale,
                file_per_band=False,
                crs=crs, 
                region = bbox_geom
            )
    
            print(f"Downloaded Image: {image_id}_{tile_id}_{scale}m.tif")
        else:
            print(f"{filename} Already Exists")