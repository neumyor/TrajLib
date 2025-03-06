import pandas as pd
import geopandas as gpd
import json
from tqdm import tqdm
from trajlib.data_processing.utils.data_definition import TrajectoryData


def read_porto_csv(csv_filepath, row_limit=None):
    csv_data = pd.read_csv(
        csv_filepath,
        on_bad_lines="warn",
        nrows=row_limit,
    )

    point_counter = 0
    traj_data = TrajectoryData()
    data = []
    traj_info = []
    for index, row in tqdm(csv_data.iterrows()):
        traj_info.append({"traj_id": row["TRIP_ID"]})

        gps_loc_list = json.loads(
            row["POLYLINE"]
        )  # 解析为坐标点列表，格式通常是[[lng, lat], [lng, lat], ...]

        timestamp_list = [
            (row["TIMESTAMP"] + 15 * i) * 1000000000 for i in range(len(gps_loc_list))
        ]  # 每个时间戳间隔15秒, *1000000000是为了符合keplergl和tbd的调包要求

        id_list = [
            row["TRIP_ID"] for _ in range(len(gps_loc_list))
        ]  # 所有坐标点的 TRIP_ID 相同

        lng_list = [loc[0] for loc in gps_loc_list]  # 获取每个坐标点的经度
        lat_list = [loc[1] for loc in gps_loc_list]  # 获取每个坐标点的纬度

        for traj_id, timestamp, lon, lat in zip(
            id_list, timestamp_list, lng_list, lat_list
        ):
            data.append(
                {
                    "point_id": point_counter,
                    "timestamp": timestamp,
                    "traj_id": traj_id,
                    "lon": lon,
                    "lat": lat,
                }
            )
            point_counter += 1

        if len(id_list) == 0:
            continue

    keys = ["point_id", "timestamp", "traj_id", "lon", "lat"]
    point_table = gpd.GeoDataFrame(data, columns=keys)
    traj_table = gpd.GeoDataFrame(traj_info, columns=["traj_id"])

    traj_data.point_table = pd.concat([point_table, traj_data.point_table])
    traj_data.traj_table = pd.concat([traj_table, traj_data.traj_table])

    return traj_data