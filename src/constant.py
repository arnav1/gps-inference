# Config

## Filepaths
# Replace with the appropriate file paths on your local machine
MAZ_FILEPATH = "path/to/maz.shp"
LAND_USE_FILEPATH = "path/to/land_use.shp"
DATA_LOC = "path/to/processed/{0}.csv"

## Column names

### From the data
# These are column names expected in the input data
TIME_COL = "CaptureDate"
LAT_COL = "Latitude"
LONG_COL = "Longitude"
DEVICE_COL = "DeviceId"
WEIGHT_COL = "weight_class"
PROVIDER_COL = "provider_id"
LAND_USE_COL = "LU15CATN"

### Set by the user
# These are column names that will be used in the processed data
DISTANCE_COL = "Distance"
DURATION_COL = "Duration"
D_TOUR_COL = "d_tour"
EVENT_COL = "Label"
HUB_COL = "hub"
TOUR_COL = "tour"

## Miscellaneous
# Replace with the appropriate provider ID
MATCHING_PROVIDER = "your_provider_id"
LOCAL_PROJ = "epsg:2262"

# Used in the code
# MONTHS: A mapping from numeric month to its string representation
MONTHS = {
    1: "january",
    4: "april",
    7: "july",
    10: "october",
}
# WEIGHT_MAP: A mapping from numeric weight class to its string representation
WEIGHT_MAP = {
    1: "Light",
    2: "Medium",
    3: "Heavy",
}
# LAND_USE_DICT: A dictionary defining valid and invalid land uses
LAND_USE_DICT = {
    "Valid": [ # The following land use must exist in the land use file
        "Residential",
        "Industrial",
        "Transportation",
        "Utility",
        "Commercial",
        "Institutional",
        "Military",
        "Recreation",
        "Agriculture",
        "Mining",
    ],
    "Invalid": [ # The following land use must exist in the land use file
        "Highway ROW",
        "Wooded",
        "Water",
        "Undeveloped",
        "Unknown",
    ],
}
# MAZ_CORRECTION: A constant to avoid maz ID collision with distance cluster IDs
MAZ_CORRECTION = 1e6
