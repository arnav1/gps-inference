import pyproj
import pandas as pd
import numpy as np

from activity import haversine_np
from constant import *


def _find_ols_bearing(data, transformer=None):
    """
    Private function to find the bearing using the slope of the least squares
    fit line with an option to transform coordinates.
    """
    if not transformer:
        x, y = transformer.transform(data[LONG_COL], data[LAT_COL])
    else:
        x, y = data[LONG_COL].values, data[LAT_COL].values
    m = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return ((np.pi / 2) - np.arctan(m)) % (2 * np.pi)


def _find_device_bearing(data, travel_time=5, initial=True, ols=False):
    """
    Private function to find the initial bearing of devices after some travel
    time.
    `travel_time` is in minutes.
    `initial` flag determines if the bearing is extracted using the initial
        part of trajectory or end
    `ols` flag will use the slope from OLS regression-fit line to find the
        bearing
    """
    # implement travel time mask
    if initial:
        keep = "first"
    else:
        keep = "last"

    time_mask = data[TIME_COL] - data.groupby(DEVICE_COL)[
        TIME_COL
    ].transform(keep)
    time_mask = time_mask.dt.total_seconds()

    if initial:
        time_mask = time_mask <= travel_time * 60
    else:
        time_mask = time_mask >= -1 * travel_time * 60

    # find bearing
    if not ols:
        bearings = (
            data.loc[time_mask]
            .groupby(DEVICE_COL)
            .agg({LAT_COL: ["first", "last"], LONG_COL: ["first", "last"]})
            .reset_index()
        )
        geodesic = pyproj.Geod(ellps="WGS84")
        bearings["bearing"], _, _ = geodesic.inv(
            bearings[LONG_COL]["first"],
            bearings[LAT_COL]["first"],
            bearings[LONG_COL]["last"],
            bearings[LAT_COL]["last"],
        )
        bearings["bearing"] = np.radians(bearings["bearing"])

        # clean up
        bearings = bearings[[DEVICE_COL, "bearing"]]
        bearings.columns = bearings.columns.droplevel(1)
    else:
        transformer = pyproj.Transformer.from_crs("epsg:4326", LOCAL_PROJ)
        bearings = (
            data.loc[time_mask]
            .groupby(DEVICE_COL)
            .apply(_find_ols_bearing, transformer=transformer)
        )

        # clean up
        bearings = bearings.to_frame("bearing").reset_index()

    return bearings


def _find_device_inferred_speed(data):
    """
    Private function to find the inferred speed of device in miles per hour.
    """
    # get the required information
    inferred_speed = (
        data[[DEVICE_COL, LAT_COL, LONG_COL, TIME_COL]]
        .groupby(DEVICE_COL)
        .nth([1, 2])
        .reset_index()
    )

    # calculate the speed
    inferred_speed["distance"] = (
        haversine_np(
            inferred_speed[LAT_COL],
            inferred_speed[LONG_COL],
            inferred_speed[LAT_COL].shift(1),
            inferred_speed[LONG_COL].shift(1),
        )
        * 1000
    )  # km to m
    inferred_speed["time"] = (
        inferred_speed[TIME_COL].diff().dt.total_seconds()
    )
    inferred_speed.drop_duplicates(subset=[DEVICE_COL], keep="last", inplace=True)
    inferred_speed["inferred_speed"] = (
        inferred_speed["distance"] / inferred_speed["time"]
    )
    inferred_speed["inferred_speed"] *= 2.237  # meter/sec to miles per hour

    # clean up
    inferred_speed = inferred_speed[[DEVICE_COL, "inferred_speed"]]

    return inferred_speed


def _match_device(row, candidate_table):
    """
    Private function that matches a row containing an eligible device to
    candidates from the candidate table.

    It implements the foloowing filters:
    1. Filter 1: Transmission time filter (within 15 minutes)
    2. Filter 2: Membership filter (within radius of 70 mph * 15 minutes = 28.16352 km)
    3. Filter 3: Direction of travel filter (within $\pm$ 45 degrees)
    4. Filter 4: Weight class filter (must be the same)
    5. Filter 5: Inferred speed filter (inferred and implied speed should be within 10 mph of each other)
    6. Filter 6: Implied speed filter (implied speed should be less than 60 miles per hour)

    And finally, if there is exactly one match, that is selected.
    """
    # filter 1
    times = (candidate_table["first_time"] - row["last_time"]).dt.total_seconds()
    transmission_time_mask = (times > 0) & (times < 15 * 60)

    # filter 2
    distances = haversine_np(
        candidate_table["first_lon"],
        candidate_table["first_lat"],
        np.repeat(row["last_lon"], len(candidate_table), axis=0),
        np.repeat(row["last_lat"], len(candidate_table), axis=0),
    )  # km
    membership_mask = distances <= 28.16352

    # filter 3
    direction = (candidate_table["bearing"] - row["bearing"] + 2 * np.pi) % (
        2 * np.pi
    )
    direction_mask = np.logical_and(
        direction >= 0, direction <= np.pi / 4
    ) | np.logical_and(direction >= 7 * np.pi / 4, direction <= 2 * np.pi)

    # filter 4
    weight_class_mask = candidate_table[WEIGHT_COL] == row[WEIGHT_COL]

    # filter 5
    implied_speed = (distances / times) * 2237  # kmps to mph
    inferred_speed_mask = (
        np.absolute(implied_speed - candidate_table["inferred_speed"]) <= 10
    )

    # filter 6: there is no need for a (implied_speed > 0) filter
    implied_speed_mask = implied_speed <= 60

    # combining them
    candidate_mask = (
        transmission_time_mask
        & membership_mask
        & direction_mask
        & weight_class_mask
        & inferred_speed_mask
        & implied_speed_mask
    )

    # check for unique match
    if candidate_mask.sum() != 1:
        result = 0
    else:
        result = candidate_table[DEVICE_COL].values[candidate_mask][0]

    return result


def _create_synthetic_ids(n=10, size=5):
    """
    Private function to create synthetic IDs
    n: Number of IDs to create
    size: Length of each ID
    """
    from random import choice
    from string import ascii_uppercase, digits

    # flag to create IDs if there are non-unique IDs
    flag = True

    while flag:
        ids = [
            "".join(choice(ascii_uppercase + digits) for i in range(size))
            for j in range(n)
        ]

        # check if the generated IDs are unique
        if len(ids) == len(set(ids)):
            flag = False

    return ids


def create_candidate_info_table(data, ols=False):
    """
    This function creates an information table for the candidates for matching.

    The information provided is based on the filters as follows:
    1. Filter 1: First time of candidate
    2. Filter 2: First location of candidate
    3. Filter 3: Initial bearing of candidate
    4. Filter 4: Weight class of candidate
    5. Filter 5: Inferred speed of candidate, implied speed (location, time)
    6. Filter 6: Implied speed of candidation
    """
    # add basic information for filters in inline comments
    candidate_table = (
        data.groupby(DEVICE_COL)
        .agg(
            {
                TIME_COL: "first",  # 1
                LAT_COL: "first",  # 2, 5 (implied), 6
                LONG_COL: "first",  # 2, 5 (implied), 6
                WEIGHT_COL: "first",  # 4
            }
        )
        .reset_index()
    )
    candidate_table.rename(
        columns={
            TIME_COL: "first_time",
            LAT_COL: "first_lat",
            LONG_COL: "first_lon",
        },
        inplace=True,
    )

    # add the bearing
    candidate_table = pd.merge(
        candidate_table,
        _find_device_bearing(data, ols=ols),
        how="left",
        on=DEVICE_COL,
    )

    # add the inferred speed
    candidate_table = pd.merge(
        candidate_table,
        _find_device_inferred_speed(data),
        how="left",
        on=DEVICE_COL,
    )

    return candidate_table


def create_eligible_info_table(data, ols=False):
    """
    This function creates an information table for the eligible devices for
    matching.

    The filters need following information for eligible devices:
    1. Filter 1: Last time of eligible device
    2. Filter 2: Last location of eligible device
    3. Filter 3: Final bearing of eligible device
    4. Filter 4: Weight class of eligible device
    5. Filter 5: Implied speed (location, time)
    6. Filter 6: Implied speed of eligible device
    """
    # add basic information for filters in inline comments
    eligible_table = (
        data.groupby(DEVICE_COL)
        .agg(
            {
                TIME_COL: "last",  # 1
                LAT_COL: "last",  # 2, 5, 6
                LONG_COL: "last",  # 2, 5, 6
                WEIGHT_COL: "last",  # 4
            }
        )
        .reset_index()
    )
    eligible_table.rename(
        columns={
            TIME_COL: "last_time",
            LAT_COL: "last_lat",
            LONG_COL: "last_lon",
        },
        inplace=True,
    )

    # add the bearing
    eligible_table = pd.merge(
        eligible_table,
        _find_device_bearing(data, initial=False, ols=ols),
        how="left",
        on=DEVICE_COL,
    )

    return eligible_table


def matching_manager(eligible_table, candidate_table, hour=23, minute=45):
    """
    This function manages the overall mathcing process by appling filters

    Only eligible devices whose last time of transmission lies after
        `hour:minute` are matched.
    """

    hour_mask = eligible_table["last_time"].dt.hour == hour
    minute_mask = eligible_table["last_time"].dt.minute >= minute
    mask = hour_mask & minute_mask
    print(f"Found {mask.sum()} eligible devices for matching.")

    eligible_table.drop(eligible_table[~mask].index, inplace=True)
    eligible_table["matched_device"] = eligible_table.apply(
        lambda row: _match_device(row, candidate_table), axis=1
    )

    print(
        f"Found matches for {(eligible_table['matched_device']!=0).sum()} devices."
    )

    return eligible_table


def create_synthetic_dataset(data, reset_time=14):
    """
    Function to generate a synthetic dataset.

    Returns a tuple of (device_summary, synthetic_dataset)
    where the `device_summary` dataframe contains the ground truth of the device ID resets.
    """
    # at least have 50 data points at 1-2 pm
    time_counts = (
        data.loc[data[TIME_COL].dt.hour == reset_time - 1]
        .groupby(DEVICE_COL)
        .size()
        .to_frame("count")
    )
    time_counts = time_counts[time_counts["count"] >= 50]

    # at least have 50 data points at 2-3 pm and previous condition
    time_counts = (
        data.loc[
            (data[DEVICE_COL].isin(time_counts.index))
            & (data[TIME_COL].dt.hour == reset_time)
        ]
        .groupby(DEVICE_COL)
        .size()
        .to_frame("count")
    )
    time_counts = time_counts[time_counts["count"] >= 50]

    # count total time of device persistance
    device_summary = (
        data.loc[data[DEVICE_COL].isin(time_counts.index)]
        .groupby(DEVICE_COL)[TIME_COL]
        .agg(lambda x: np.ptp(x).total_seconds())
        .to_frame("total_duration")
    )

    # remove devices less than 8 hours and more than 16 hours durations
    device_summary = device_summary[device_summary["total_duration"] > 8 * 3600]
    device_summary = device_summary[device_summary["total_duration"] < 16 * 3600]
    device_summary.drop(columns=["total_duration"], inplace=True)

    device_summary["synthetic_id"] = _create_synthetic_ids(n=len(device_summary))

    # get the synthetic dataset
    synthetic = data.loc[data[DEVICE_COL].isin(device_summary.index)].copy()
    synthetic.reset_index(drop=True, inplace=True)
    synthetic = pd.merge(
        synthetic,
        device_summary,
        how="left",
        left_on=DEVICE_COL,
        right_index=True,
    )

    synthetic.loc[
        synthetic[TIME_COL].dt.hour > reset_time - 1, DEVICE_COL
    ] = synthetic["synthetic_id"]

    assert (
        len(set(synthetic["synthetic_id"]).difference(synthetic[DEVICE_COL])) == 0
    )
    assert len(set(device_summary.index).difference(synthetic[DEVICE_COL])) == 0

    return device_summary, synthetic
