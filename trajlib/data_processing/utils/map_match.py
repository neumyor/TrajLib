import transbigdata as tbd
import pandas as pd
import os
import osmnx as ox
import re
import geopandas as gpd
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
from .data_definition import TrajectoryData, GeoData, GeoRelationData


def get_roadnetwork(bounds, cache_dir, network_type="drive"):
    bounds_name = "_".join([str(num).replace(".", "-") for num in bounds])
    cache_filepath = os.path.join(
        cache_dir, f"cache_{network_type}_{bounds_name}.graphml"
    )
    if os.path.exists(cache_filepath):
        G = ox.load_graphml(cache_filepath)
    else:
        G = ox.graph_from_bbox(bounds, network_type=network_type)
    ox.save_graphml(G, cache_filepath)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    map_con = InMemMap(
        name="pNEUMA",
        use_latlon=True,
        use_rtree=True,
        index_edges=True,
    )  # , use_rtree=True, index_edges=True)

    for node_id, row in nodes.iterrows():
        map_con.add_node(node_id, (row["y"], row["x"]))
    for node_id_1, node_id_2, _ in G.edges:
        map_con.add_edge(node_id_1, node_id_2)

    return map_con, nodes, edges


import time


def match_traj_data_with_roadnetwork(
    traj_data: TrajectoryData, map_con: InMemMap, nodes, edges
):
    traj_list = traj_data.cal_all_trajs(attrs=["point_id", "lon", "lat"])
    traj_states_list = []
    all_states = []

    from tqdm import tqdm

    # Phase 1: Trajectory Matching
    print("Starting trajectory matching...")
    start_time = time.time()
    for traj_id in tqdm(traj_list.keys()):
        attrs_value = traj_list[traj_id]
        path = [
            tuple([lon, lat])
            for lon, lat in zip(attrs_value["lon"], attrs_value["lat"])
        ]
        states_list, _ = match_gps_path_with_roadnetwork(
            lon_lat_list=path, map_con=map_con
        )
        traj_states_list.append(states_list)
    end_time = time.time()
    print(f"Trajectory matching completed, time taken: {end_time - start_time} seconds")

    # Phase 2: Processing Matching States
    print("Starting processing matching states...")
    start_time = time.time()
    for states_list in traj_states_list:
        states_list = [(state[0], state[1]) for state in states_list]
        all_states.extend(set(states_list))
    all_states = list(set(all_states))
    road2id, road_connection = assign_edge_id_and_connections(all_states)
    end_time = time.time()
    print(
        f"Processing matching states completed, time taken: {end_time - start_time} seconds"
    )

    # Phase 3: Updating Point Table
    print("Starting updating point table...")
    start_time = time.time()
    point_id_list_list = [record['point_id'] for record in traj_list.values()]
    for point_id_list, states_list in zip(point_id_list_list, traj_states_list):
        road_ids = [road2id[state] for state in states_list]
        for point_id, road_id in zip(point_id_list, road_ids):
            traj_data.point_table.loc[
                traj_data.point_table["point_id"] == point_id, "road_id"
            ] = road_id
    end_time = time.time()
    print(
        f"Updating point table completed, time taken: {end_time - start_time} seconds"
    )

    # Phase 4: Creating Geo Data
    print("Starting creating geo data...")
    start_time = time.time()
    geo_data = GeoData()
    info_table_data = []
    for edge, edge_id in road2id.items():
        matched_edge = edges.loc[edge]
        if not matched_edge.empty:
            line_string = matched_edge["geometry"].iloc[0]
        else:
            print(f"No road information found for node pair {edge}")

        info_table_data.append(
            {
                "geo_id": edge_id,
                "type": "road",
                "coord": line_string,
            }
        )
    for row in info_table_data:
        geo_data.append_info_data(row)
    end_time = time.time()
    print(f"Creating geo data completed, time taken: {end_time - start_time} seconds")

    # Phase 5: Creating Geo Relation Data
    print("Starting creating geo relation data...")
    start_time = time.time()
    geo_relation_data = GeoRelationData()
    relation_table_data = []
    rel_id_counter = 0
    for edge_id_1, edge_id_2 in road_connection:
        relation_table_data.append(
            {"rel_id": rel_id_counter, "origin_id": edge_id_1, "dest_id": edge_id_2}
        )
        rel_id_counter += 1
    geo_relation_data.relation_table = gpd.GeoDataFrame(
        relation_table_data, columns=geo_relation_data.essential_relation_attr
    )
    end_time = time.time()
    print(
        f"Creating geo relation data completed, time taken: {end_time - start_time} seconds"
    )

    print("All phases completed")
    return traj_data, geo_data, geo_relation_data


def assign_edge_id_and_connections(states: list):
    """
    根据输入的states，将其中涉及的道路段分配一个id，并且建立起他们的关联
    如果一个道路段的终点，恰好是另一个道路段的起点，则两个道路相连
    #TODO：该函数应该被替换掉，采用Openstreetmap中的唯一id及其相连关系
    """
    assert len(states[0]) == 2

    edge_id_mapping = {}
    edge_id_counter = 0
    connection_table = []

    # 步骤一：为每种二元组分配一个edge_id
    for edge in states:
        edge_tuple = tuple(edge)
        if edge_tuple not in edge_id_mapping:
            edge_id_mapping[edge_tuple] = edge_id_counter
            edge_id_counter += 1

    # 步骤二：构建edge_id之间连接关系的表
    for i in range(len(states)):
        for j in range(len(states)):
            if i != j:
                current_edge = states[i]
                other_edge = states[j]
                if current_edge[1] == other_edge[0]:
                    edge_id_1 = edge_id_mapping[tuple(current_edge)]
                    edge_id_2 = edge_id_mapping[tuple(other_edge)]
                    connection_table.append((edge_id_1, edge_id_2))

    return edge_id_mapping, connection_table


def match_gps_path_with_roadnetwork(
    lon_lat_list: list[tuple],
    map_con: InMemMap,
    visualization=False,
):
    """
    返回的state包含多个二元组，每个二元组代表着一个道路段
    例如(123, 345)可以看作从起点为123终点为345的一个道路段
    """
    path = [tuple([lat, lon]) for (lon, lat) in lon_lat_list] # 这个包要求的经纬度要反过来

    # 构建地图匹配工具
    matcher = DistanceMatcher(
        map_con,
        max_dist=300,
        max_dist_init=250,
        min_prob_norm=0.0001,
        non_emitting_length_factor=0.95,
        obs_noise=50,
        obs_noise_ne=50,
        dist_noise=50,
        max_lattice_width=20,
        non_emitting_states=True,
    )

    # 进行地图匹配
    states, _ = matcher.match(path, unique=False)

    if visualization:
        mmviz.plot_map(
            map_con,
            matcher=matcher,
            show_labels=False,
            show_matching=True,
            filename=None,
            figwidth=5,
        )
    return states, matcher
