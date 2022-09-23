import dask.dataframe as dd
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from constant import *


def _apply_aggregation_filter(data, cumsum_groups, keep_first=True):
    """
    Private function that essentially takes a cumsum based grouping to
    aggregate rows together. The effect is that the rows are 'filtered' out
    of the dataset while adding the device distances and durations.

    By default, the function will keep the first row in a group consecutive
    rows.
    """
    # check if we need to return a geodataframe
    is_gdf = isinstance(data, gpd.GeoDataFrame)
    if is_gdf:
        gdf_crs = data.crs

    # create the aggregation dictionary
    if keep_first:
        keep = "first"
    else:
        keep = "last"
    aggs = {c: keep for c in data.columns}
    aggs[DURATION_COL] = "sum"
    aggs[DISTANCE_COL] = "sum"

    data = data.groupby(cumsum_groups).agg(aggs)
    data = data.reset_index(drop=True)
    if is_gdf:
        data = gpd.GeoDataFrame(data, geometry="geometry", crs=gdf_crs)

    return data


def _apply_min_event_filter(data, col=D_TOUR_COL, num_min_events=1):
    """
    Private function to remove `d_tours` that have fewer than `num_min_events`
    number of `starting` and `stopping` events.
    """
    event_counts = data.groupby([col, EVENT_COL]).size().unstack(fill_value=0)
    d_tours_to_keep = event_counts.index[
        (event_counts["starting"] > num_min_events)
        & (event_counts["stopping"] > num_min_events)
    ]
    data = data.loc[data[D_TOUR_COL].isin(d_tours_to_keep.values)]

    data.reset_index(drop=True, inplace=True)
    return data


def _assign_nan_to_first(data, col=D_TOUR_COL):
    """
    Private function to assign a NaN/NaT value to `Distance`/`Duration`
    field respectively of the first row using groups based on the column `col`.
    """
    mask = data[col] != data[col].shift(1)
    data.loc[mask, DURATION_COL] = np.nan
    data.loc[mask, DISTANCE_COL] = np.nan
    return data


def _fix_d_tour_top(data):
    """
    Private function to:
    1. Remove `stopping` events from the tops of `d_tours`
    2. Set the Duration/Distance of first `starting` event to NaN
    3. Reset index
    """
    # step 1
    first_mask = data.index.isin(data.groupby(D_TOUR_COL).head(1).index)
    stopping_mask = (data[EVENT_COL] == "stopping").values
    mask = (first_mask ^ stopping_mask) | (~stopping_mask)
    data = data[mask]

    # step 2
    data = _assign_nan_to_first(data)

    # step 3
    data.reset_index(drop=True, inplace=True)

    return data


def _fix_d_tour_bottom(data):
    """
    Private function to:
    1. Remove `starting` events from the bottom of `d_tours`
    2. Reset index
    """
    # step 1
    last_mask = data.index.isin(data.groupby(D_TOUR_COL).tail(1).index)
    starting_mask = (data[EVENT_COL] == "starting").values
    mask = (last_mask ^ starting_mask) | (~starting_mask)
    data = data[mask]

    # step 2
    data.reset_index(drop=True, inplace=True)

    return data


def _init_gdf(data):
    """
    Function takes the `data` table and converts it into the geo-equivalent.
    """
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.Longitude, data.Latitude),
        crs="EPSG:4326",
    )


def _create_agglomerative_clusters(geom, threshold=500, **kwargs):
    """
    Private function to create cluster through agglomerative clustering using
    `complete` linkage. The threshold is in feet.

    1. It selects the `stopping` events and finds their cluster membership
    2. Cleanup:
        1. Every `starting` event following a `stopping` event has the same
           cluster membership.
        2. The first `starting` event has cluster number -1
    """
    # step 1
    geom = np.vstack(geom)[1::2]
    D = pdist(geom, metric="euclidean")  # in ft
    result = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        linkage="complete",
        distance_threshold=threshold,
    ).fit(squareform(D))

    # step 2
    result = np.repeat(result.labels_, 2, axis=0)
    result = np.roll(result, shift=1, axis=0)
    result[0] = -1

    return result


def _create_dbscan_clusters(geom, eps=500, **kwargs):
    """
    Private function to create cluster through DBSCAN clustering.

    1. It selects the `stopping` events and finds their cluster membership
    2. Cleanup:
        1. Every `starting` event following a `stopping` event has the same
           cluster membership.
        2. The first `starting` event has cluster number -1
    """
    # step 1
    geom = np.vstack(geom)[1::2]
    D = pdist(geom, metric="euclidean")  # in ft
    result = DBSCAN(
        eps=eps, min_samples=1, algorithm="auto", metric="precomputed"
    ).fit(squareform(D))

    # step 2
    result = np.repeat(result.labels_, 2, axis=0)
    result = np.roll(result, shift=1, axis=0)
    result[0] = -1

    return result


def load_data(fmt=1, matching=False, drop_threshold=10, time_threshold=10):
    """
    Function loads the data and does the following:
    1. convert date column to correct format
    2. sort values by ['DeviceId', 'CaptureDate']

    drop_threshold is the minimum number of records needed to keep a device
    time_threshold is the minimum device duration in minutes to keep a device
    """
    # find the file/s to load
    if isinstance(fmt, int):
        file_loc = DATA_LOC.format(MONTHS[fmt])
    elif isinstance(fmt, str):
        file_loc = DATA_LOC.format(fmt)

    # load the data
    data = dd.read_csv(file_loc, parse_dates=[0])
    if matching:
        data = data[(data[PROVIDER_COL] == MATCHING_PROVIDER)]
    data = data.compute()

    # drop devices with less number of records or device duration
    data = data[
        data.groupby(DEVICE_COL)[DEVICE_COL].transform("size") > drop_threshold
    ]
    data = data[
        data.groupby(DEVICE_COL)[TIME_COL].transform(
            lambda x: np.ptp(x).total_seconds()
        )
        > time_threshold * 60
    ]

    # sort values for everything
    data.sort_values([DEVICE_COL, TIME_COL], inplace=True, ignore_index=True)

    return data


def assign_labels(data, threshold=6, baseline=False):
    """
    This function takes in raw data and returns labels based on a speed
    threshold.
    If baseline is True, then the d_tour is basically device ID.

    The threshold is provided in miles per hour.
    """
    # mask for correction at the boundary/find d_tour
    device_mask = data[DEVICE_COL] != data[DEVICE_COL].shift(1)
    data["TimeFromPrev"] = data[TIME_COL].diff()
    data["TimeFromPrev"] = data["TimeFromPrev"].dt.total_seconds()
    if not baseline:
        duration_mask = data["TimeFromPrev"] > 8 * 60 * 60
    else:
        duration_mask = np.zeros_like(device_mask).astype(bool)
    mask = np.logical_or(device_mask, duration_mask)  # d_tour mask

    # time
    data.loc[mask, "TimeFromPrev"] = np.nan
    data["TimeToNext"] = data["TimeFromPrev"].shift(-1)

    # distance
    data["DistanceFromPrev"] = haversine_np(
        data[LAT_COL],
        data[LONG_COL],
        data[LAT_COL].shift(1),
        data[LONG_COL].shift(1),
    )
    data.loc[mask, "DistanceFromPrev"] = np.nan
    data["DistanceToNext"] = data["DistanceFromPrev"].shift(-1)

    # speed
    data["SpeedFromPrev"] = data["DistanceFromPrev"] / data["TimeFromPrev"]
    data["SpeedFromPrev"] *= 2236.94  # kmps to mph
    data["SpeedToNext"] = data["DistanceToNext"] / data["TimeToNext"]
    data["SpeedToNext"] *= 2236.94  # kmps to mph

    # assign label
    data["SpeedPrevMask"] = (data["SpeedFromPrev"] < threshold).astype(int)
    data["SpeedNextMask"] = (data["SpeedToNext"] < threshold).astype(int)
    df = pd.DataFrame(
        [
            [1, 1, "stopped"],
            [0, 0, "moving"],
            [1, 0, "starting"],
            [0, 1, "stopping"],
        ],
        columns=["SpeedPrevMask", "SpeedNextMask", EVENT_COL],
    )
    data = data.merge(df, how="left")

    # save the d_tour number
    data[D_TOUR_COL] = mask.cumsum()

    # cleanup dataframe
    remove_cols = [
        "TimeToNext",
        "DistanceToNext",
        "SpeedFromPrev",
        "SpeedToNext",
        "SpeedNextMask",
        "SpeedPrevMask",
    ]
    reuse_cols = {
        "TimeFromPrev": DURATION_COL,
        "DistanceFromPrev": DISTANCE_COL,
    }
    data.drop(columns=remove_cols, inplace=True)
    data.rename(columns=reuse_cols, inplace=True)
    data[DISTANCE_COL] *= 1000  # km to m

    return data


def find_distance_duration(data):
    """
    This function does the following:
    1. Trim `moving` events from top and bottom
    2. Aggregate `moving` and `stopped` in between
    3. Fix the top and bottom of d_tours
    4. Remove d-tours without at least 2 starting and stopping events eachs
    5. Cleanup: remove columns created here
    """
    # step 1
    data["desirable_mask"] = data[EVENT_COL].map(
        {"moving": 0, "stopped": 0, "stopping": 1, "starting": 1}
    )
    forward_mask = data.groupby(D_TOUR_COL)["desirable_mask"].cumsum().ne(0)
    backward_mask = (
        data[::-1].groupby(D_TOUR_COL)["desirable_mask"].cumsum().ne(0)[::-1]
    )
    mask = forward_mask & backward_mask
    data.drop(data[~mask].index, inplace=True)

    # step 2
    data = _apply_aggregation_filter(
        data,
        data["desirable_mask"].shift(1, fill_value=True).cumsum(),
        keep_first=False,
    )

    # step 3
    data = _fix_d_tour_top(data)
    data = _fix_d_tour_bottom(data)

    # step 4
    data = _apply_min_event_filter(data)

    # step 5
    data.drop(columns=["desirable_mask"], inplace=True)

    return data


def apply_stop_duration_filter(data, duration=3):
    """
    Function to apply the stop duration filter on raw data. See
    `apply_stop_duration` for operations on delivered data.

    `duration` kwarg is in minutes. The `Duration` column is
    expected to be in seconds.
    """
    duration *= 60  # min to sec

    # apply stop duration
    event_mask = (data[EVENT_COL] == "starting") & (data[DURATION_COL] < duration)
    event_mask = ~(event_mask | event_mask.shift(1, fill_value=False))
    data = _apply_aggregation_filter(data, event_mask.cumsum(), keep_first=False)

    # cleanup: filter d-tours and add nan's
    data = _fix_d_tour_top(data)
    data = _fix_d_tour_bottom(data)
    data = _apply_min_event_filter(data)

    return data


def init_land_use():
    """
    Loads the land use file.
    """
    return gpd.read_file(LAND_USE_FILEPATH)[
        [LAND_USE_COL, "geometry"]
    ]


def init_maz():
    """
    Loads the MAZ file
    """
    maz_data = gpd.read_file(MAZ_FILEPATH)
    maz_data = maz_data.to_crs(epsg=4326)
    maz_data = maz_data[["ID", "geometry"]]
    maz_data = maz_data.rename(columns={"ID": "maz_cluster"})
    maz_data["maz_cluster"] = maz_data["maz_cluster"] + MAZ_CORRECTION
    return maz_data


def add_land_use(data, land_use):
    """
    Function adds land use information to dataframe.

    Since we use `LU15SUBN` as the column to do aggregations later, we make
    some temporary fixes for the paper.
    """
    if "geometry" not in data.columns:
        data = _init_gdf(data)

    data = gpd.sjoin(data, land_use, how="left", predicate="within")
    data.drop(labels=["index_right"], axis=1, inplace=True)
    data[LAND_USE_COL] = data[LAND_USE_COL].fillna("Unknown")
    return data


def add_maz(data, maz_data):
    """
    Function adds MAZ information to dataframe.
    """
    if "geometry" not in data.columns:
        data = _init_gdf(data)

    data = gpd.sjoin(data, maz_data, how="left", predicate="within")
    if "index_right" in data.columns:
        data.drop(labels=["index_right"], axis=1, inplace=True)

    return data


def add_distance_cluster(
    data, algo="dbscan", col_name="dist_cluster", threshold=500, eps=500
):
    """
    Create agglomerative clusters from geodataframe.
    `col_name` sets the name of the column which will store the cluster
        membership information.
    `threshold` and `dist` is in feet.
    `threshold` only used when algo='agglomerative'
    `eps` only used when algo='dbscan'
        (Default value from previous code is 1320 ft)
    """
    if "geometry" not in data.columns:
        data = _init_gdf(data)

    if data.crs != LOCAL_PROJ:
        data = data.to_crs(LOCAL_PROJ)  # coords in feet

    if algo == "dbscan":
        func = _create_dbscan_clusters
    elif algo == "agglomerative":
        func = _create_agglomerative_clusters
    else:
        raise NotImplementedError(
            f'Clustering algorithm "{algo}" has not been implemented.'
        )

    data["pts"] = data["geometry"].apply(lambda p: p.coords[0])
    data[col_name] = data.groupby(D_TOUR_COL, sort=False)["pts"].transform(
        func, threshold=threshold, eps=eps
    )

    # clean up
    data.drop(columns=["pts"], inplace=True)
    data = data.to_crs(epsg="4326")

    return data


def add_tour_numbers(
    data, by_col=D_TOUR_COL, maz_col="maz_cluster", dist_col="dist_cluster"
):
    """
    Function to find the hubs and assign tour numbers to dataframe.

    The hub is found using the following rules
    1. The mode of MAZs is the hub
    2. Ties are broken using total stop duration at the associated distance
       clusters
    3. Found hubs are extended using distance-based cluster memberships to
       yield the final hub

    For stop whose MAZ values are NaN. They are temporarily provided a MAZ
    value of the distance cluster they are associated with. This in essence
    creates a distance cluster size and shape `MAZ` around them. There are
    provisions to ensure actual MAZ IDs do not collide with distance
    cluster-based IDs.
    """
    # warming up
    data.sort_values([by_col, TIME_COL], inplace=True, ignore_index=True)

    # let us separate the stuff so that the original dataframe is maintained
    df = data.loc[data[EVENT_COL] == "stopping"][
        [by_col, maz_col, dist_col, DURATION_COL]
    ].copy()

    # fill in the nan values
    df[maz_col] = df[maz_col].fillna(df[dist_col]).astype(int)

    # count the maz occurances in each d_tour, will ultimately hold MAZ mode for each d_tour
    # the nan's close by are grouped together and counted
    maz_mode = (
        df.groupby([by_col, maz_col]).size().to_frame("count").reset_index()
    )

    # calculate the total time stopped at a distance cluster in each d-tour
    dist_cluster_duration = (
        df.groupby([by_col, dist_col]).agg({DURATION_COL: "sum"}).reset_index()
    )

    # finds the unique combinations of maz ID and distance cluster ID in each d-tour
    # one group of nan's cannot be connected another group of spatially separate nan's
    maz_distance = df[[by_col, maz_col, dist_col]].drop_duplicates()

    # add the duration of each distance cluster
    maz_distance = pd.merge(
        maz_distance, dist_cluster_duration, how="left", on=[by_col, dist_col]
    )

    # aggregate based on maz ID to find total stop duration for all distance clusters based in that MAZ ID
    # for the nan's, the MAZ is implicitly assumed to be the size and shape of the distance cluster
    maz_distance = (
        maz_distance.groupby([by_col, maz_col])
        .agg({DURATION_COL: "sum"})
        .reset_index()
    )

    # finally add this information to the maz_mode dataframe
    maz_mode = pd.merge(maz_mode, maz_distance, how="left", on=[by_col, maz_col])

    # sort count, tie-break by stop duration of distance cluster
    # drop duplicates keeps the first (highest) instance
    # the regular drop and sort_values cleans up the dataframe
    maz_mode = (
        maz_mode.sort_values(by=["count", DURATION_COL], ascending=False)
        .drop_duplicates(subset=[by_col])
        .drop(columns=["count", DURATION_COL])
        .sort_values([by_col])
    )

    # to extend the modes, get the unique combinations again
    maz_distance = df[[by_col, maz_col, dist_col]].drop_duplicates()

    # filter the unique maz-distance cluster combinations on mode maz ID for each d_tour
    maz_distance = pd.merge(
        maz_distance, maz_mode, how="inner", on=[by_col, maz_col]
    )
    maz_distance[HUB_COL] = 1

    # add hub from maz
    df = pd.merge(
        df,
        maz_distance.drop_duplicates(subset=[by_col, maz_col]),
        how="left",
        on=[by_col, maz_col],
    )
    df[HUB_COL] = df[HUB_COL].notna().astype(int)
    df.rename(columns={HUB_COL: "hub_maz", dist_col + "_x": dist_col}, inplace=True)
    df.drop(columns=[dist_col + "_y"], inplace=True)

    # add extended hub from distance cluster
    df = pd.merge(
        df,
        maz_distance.drop_duplicates(subset=[by_col, dist_col]),
        how="left",
        on=[by_col, dist_col],
    )
    df[HUB_COL] = df[HUB_COL].notna().astype(int)
    df.rename(columns={HUB_COL: "hub_dist", maz_col + "_x": maz_col}, inplace=True)
    df.drop(columns=[maz_col + "_y"], inplace=True)

    # create a hub mask
    df[HUB_COL] = df["hub_maz"] | df["hub_dist"]

    # add the zeros for the `starting` events
    hub_mask = df[HUB_COL].values
    output = np.zeros(len(hub_mask) * 2, dtype=int)
    output[1::2] = hub_mask

    # assign hub
    data[HUB_COL] = output

    # assign tour
    hub_mask = data[HUB_COL].astype(bool).shift(1, fill_value=False)
    device_mask = data[by_col].shift(1) != data[by_col]
    mask = hub_mask | device_mask
    data[TOUR_COL] = mask.cumsum()

    return data


def apply_land_use_filter(data):
    """
    Removes stops associated with land uses deemed invalid in the
    `constant.py` file. The `Unknown` land use is not droppped.
    """
    remove_list = [x for x in LAND_USE_DICT["Invalid"] if x != "Unknown"]

    d_tour_mask = data[D_TOUR_COL] != data[D_TOUR_COL].shift(1)
    land_use_mask = (data[EVENT_COL] == "stopping") & (
        data[LAND_USE_COL].isin(remove_list)
    )
    land_use_mask = ~(land_use_mask | land_use_mask.shift(1, fill_value=False))
    mask = land_use_mask | d_tour_mask

    data = _apply_aggregation_filter(data, mask.cumsum(), keep_first=True)

    # clean up
    data = _fix_d_tour_top(data)
    data = _fix_d_tour_bottom(data)
    data = _apply_min_event_filter(data)

    return data


def apply_cluster_filter(data, col_name):
    """
    Clusters stops with same cluster ID. That means that if consecutive stops
    are in the same cluster, they are assumed to be the same stop.
    """
    cluster_mask = data.loc[data[EVENT_COL] == "stopping"][col_name]
    cluster_mask = cluster_mask != cluster_mask.shift(1)
    d_tour_mask = data.loc[data[EVENT_COL] == "stopping"][D_TOUR_COL]
    d_tour_mask = d_tour_mask != d_tour_mask.shift(1)
    cluster_mask = cluster_mask | d_tour_mask
    cluster_mask = np.repeat(cluster_mask.values, 2, axis=0)

    data = _apply_aggregation_filter(data, cluster_mask.cumsum(), keep_first=True)

    # clean up
    data = _fix_d_tour_top(data)
    data = _fix_d_tour_bottom(data)
    data = _apply_min_event_filter(data)

    return data


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def apply_stop_duration(df, duration=3):
    """
    Applies the stop duration filter to fully processed dataset.
    """
    df = df.loc[~((df["EventType"] == "stopping") & (df[DURATION_COL] < duration))]
    unique_events = (df["EventType"] != df["EventType"].shift(1)).cumsum()
    return df.loc[unique_events.shift(1) != unique_events]  # keep first
