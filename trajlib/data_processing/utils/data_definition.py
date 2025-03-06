import pandas as pd
import geopandas as gpd
import os

import pickle
import geopandas as gpd
import os


class TrajectoryData:
    def __init__(self):
        self.essential_point_attr = ["point_id", "traj_id", "timestamp"]
        self.essential_traj_attr = ["traj_id"]
        self.point_table = gpd.GeoDataFrame(columns=self.essential_point_attr)
        self.traj_table = gpd.GeoDataFrame(columns=self.essential_traj_attr)

    def read_from_file(self, point_file_path, traj_file_path) -> None:
        """
        从文件读取轨迹数据并转换为内部的GeoDataframe格式
        :param point_file_path: 轨迹点表文件路径
        :param traj_file_path: 轨迹表文件路径
        """
        self.point_table = gpd.read_file(
            point_file_path
        )  # 使用gpd.read_file，假设是支持的地理数据文件格式
        self.traj_table = gpd.read_file(traj_file_path)

    def save_to_pkl(self, save_path) -> None:
        """
        将当前对象保存到一个.pkl文件中
        :param save_path: 输出文件路径
        """
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pkl(load_path) -> None:
        """
        从.pkl文件中加载对象
        :param load_path: 输入文件路径
        """
        with open(load_path, "rb") as f:
            loaded_obj = pickle.load(f)
        return loaded_obj

    def write_to_file(self, save_path) -> None:
        """
        将内部的轨迹点表和轨迹表数据写入文件，处理可选列情况
        :param point_file_path: 轨迹点表输出文件路径
        :param traj_file_path: 轨迹表输出文件路径
        """
        self.point_table.to_csv(os.path.join(save_path, "points.csv"), index=False)
        self.traj_table.to_csv(os.path.join(save_path, "trajectories.csv"), index=False)

    def append_point_data(self, new_point_data: dict, extra_attr: dict = None) -> None:
        """
        向轨迹点表追加新数据，处理可选列情况
        :param new_point_data: 新的轨迹点数据，类似GeoDataframe结构
        :param extra_attr: 可选列及对应值的字典，默认为None
        """
        assert all(key in new_point_data for key in self.essential_point_attr)

        # 遍历extra_attr的键，如果存在当前point_table没有的列，则向当前point_table追加新的列
        if extra_attr:
            for col in extra_attr.keys():
                if col not in self.point_table.columns:
                    self.point_table[col] = None

        # 根据new_point_data和extra_attr两个字典，构建新的行并添加到point_table中。
        new_row_data = {}
        for attr in self.essential_point_attr:
            new_row_data[attr] = new_point_data.get(attr)
        if extra_attr:
            for col, value in extra_attr.items():
                new_row_data[col] = value

        new_row_df = gpd.GeoDataFrame([new_row_data], columns=self.point_table.columns)
        self.point_table = gpd.GeoDataFrame(
            pd.concat([self.point_table, new_row_df], ignore_index=True)
        )

    def batch_append_point_data(
        self, new_point_data_list: list[dict], extra_attr_list: list[dict] = None
    ) -> None:
        """
        向轨迹点表批量追加新数据，处理可选列情况
        :param new_point_data: 新的轨迹点数据，类似GeoDataframe结构
        :param extra_attr: 可选列及对应值的字典，默认为None
        """
        assert all(key in new_point_data_list[0] for key in self.essential_point_attr)

        # 假设extra_attr中的每个字典拥有的键一样
        if extra_attr_list:

            assert len(new_point_data_list) == len(extra_attr_list)

            for col in extra_attr_list[0].keys():
                if col not in self.point_table.columns:
                    self.point_table[col] = None

        # 根据new_point_data和extra_attr两个字典，构建新的行并添加到point_table中。
        new_data_list = []
        for index, new_point_data in enumerate(new_point_data_list):
            new_row_data = {}
            for attr in self.essential_point_attr:
                new_row_data[attr] = new_point_data.get(attr)
            if extra_attr_list:
                for col, value in extra_attr_list[index].items():
                    new_row_data[col] = value
            new_data_list.append(new_row_data)

        new_row_df = gpd.GeoDataFrame(new_data_list, columns=self.point_table.columns)
        self.point_table = gpd.GeoDataFrame(
            pd.concat([self.point_table, new_row_df], ignore_index=True)
        )

    def cal_all_trajs(self, attrs: list = ["point_id"]):
        """
        遍历traj_table，对于每个traj_id找到point_table中所有的对应的记录，并按照timestamp排序。
        最终返回一个字典，其中key是traj_id，value是对应的按照timestamp排序后的point_id列表。
        """
        result_dict = {}
        for traj_id in self.traj_table["traj_id"].unique():
            # 根据当前traj_id筛选出point_table中对应的记录，并按照timestamp排序
            sorted_points = self.point_table[
                self.point_table["traj_id"] == traj_id
            ].sort_values(by="timestamp")
            attrs_value = {}
            for attr in attrs:
                assert attr in sorted_points.columns
                attrs_value[attr] = sorted_points[attr].tolist()

            result_dict[traj_id] = attrs_value
        return result_dict


class GeoData:
    def __init__(self):
        self.essential_info_attr = ["geo_id", "type", "coord"]
        self.info_table = gpd.GeoDataFrame(columns=self.essential_info_attr)

    def read_from_file(self, file_path):
        """
        从文件读取地理数据信息表并转换为内部的GeoDataframe格式
        :param file_path: 信息表文件路径
        """
        self.info_table = gpd.read_file(file_path)

    def write_to_file(self, save_path):
        """
        将内部的地理数据信息表写入文件，处理可选列情况
        :param save_path: 输出文件路径
        """
        self.info_table.to_csv(os.path.join(save_path, "geo_info.csv"), index=False)

    def append_info_data(self, new_info_data: dict, extra_attr: dict = None):
        """
        向信息表追加新数据，处理可选列情况
        :param new_info_data: 新的地理信息数据，类似GeoDataframe结构
        :param extra_attr: 可选列及对应值的字典，默认为None
        """
        assert all(key in new_info_data for key in self.essential_info_attr)

        # 遍历extra_attr的键，如果存在当前info_table没有的列，则向当前info_table追加新的列
        if extra_attr:
            for col in extra_attr.keys():
                if col not in self.info_table.columns:
                    self.info_table[col] = None

        # 根据new_info_data和extra_attr两个字典，构建新的行并添加到info_table中。
        new_row_data = {}
        for attr in self.essential_info_attr:
            new_row_data[attr] = new_info_data.get(attr)
        if extra_attr:
            for col, value in extra_attr.items():
                new_row_data[col] = value

        new_row_df = gpd.GeoDataFrame([new_row_data], columns=self.info_table.columns)
        self.info_table = gpd.GeoDataFrame(
            pd.concat([self.info_table, new_row_df], ignore_index=True)
        )


class GeoRelationData:
    def __init__(self):
        self.essential_relation_attr = ["rel_id", "origin_id", "dest_id"]
        self.relation_table = gpd.GeoDataFrame(columns=self.essential_relation_attr)

    def read_from_file(self, file_path):
        """
        从文件读取地理对象关系数据并转换为内部的GeoDataframe格式
        :param file_path: 关系表文件路径
        """
        self.relation_table = gpd.read_file(file_path)

    def write_to_file(self, save_path):
        """
        将内部的地理对象关系表数据写入文件，处理可选列情况
        :param save_path: 输出文件路径
        """
        self.relation_table.to_csv(
            os.path.join(save_path, "geo_relation.csv"), index=False
        )

    def append_relation_data(self, new_relation_data: dict, extra_attr: dict = None):
        """
        向关系表追加新数据，处理可选列情况
        :param new_relation_data: 新的地理关系数据，类似GeoDataframe结构
        :param extra_attr: 可选列及对应值的字典，默认为None
        """
        assert all(key in new_relation_data for key in self.essential_relation_attr)

        # 遍历extra_attr的键，如果存在当前relation_table没有的列，则向当前relation_table追加新的列
        if extra_attr:
            for col in extra_attr.keys():
                if col not in self.relation_table.columns:
                    self.relation_table[col] = None

        # 根据new_relation_data和extra_attr两个字典，构建新的行并添加到relation_table中。
        new_row_data = {}
        for attr in self.essential_relation_attr:
            new_row_data[attr] = new_relation_data.get(attr)
        if extra_attr:
            for col, value in extra_attr.items():
                new_row_data[col] = value

        new_row_df = gpd.GeoDataFrame(
            [new_row_data], columns=self.relation_table.columns
        )
        self.relation_table = gpd.GeoDataFrame(
            pd.concat([self.relation_table, new_row_df], ignore_index=True)
        )
