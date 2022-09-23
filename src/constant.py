# Config

## Filepaths
MAZ_FILEPATH = "/srv/data/arnav/dvrpc/maz/maz.shp"
LAND_USE_FILEPATH = "/srv/data/arnav/dvrpc/land_use/land_use.shp"
DATA_LOC = "/srv/data/arnav/dvrpc/processed/{0}.csv"

## Column names

### From the data
TIME_COL = "CaptureDate"
LAT_COL = "Latitude"
LONG_COL = "Longitude"
DEVICE_COL = "DeviceId"
WEIGHT_COL = "weight_class"
PROVIDER_COL = "provider_id"
LAND_USE_COL = "LU15CATN"

### Set by the user
DISTANCE_COL = "Distance"
DURATION_COL = "Duration"
D_TOUR_COL = "d_tour"
EVENT_COL = "Label"
HUB_COL = "hub"
TOUR_COL = "tour"

## Miscellaneous
MATCHING_PROVIDER = "45c48cce2e2d7fbdea1afc51c7c6ad26"
LOCAL_PROJ = "epsg:2262"

# Used in the code
MONTHS = {
    1: "january",
    4: "april",
    7: "july",
    10: "october",
}
WEIGHT_MAP = {
    1: "Light",
    2: "Medium",
    3: "Heavy",
}
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
MAZ_CORRECTION = 1e6  # to avoid maz ID collision with distance cluster IDs
