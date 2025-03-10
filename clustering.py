import os
import torch
import pandas as pd
import pickle
from easydict import EasyDict
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
from math import sqrt, atan2, sin


class Segment(EasyDict):
    def __init__(self, segment_id, points, emb):
        super().__init__()
        self.id = segment_id
        self.points = points
        self.emb = emb


class Cluster:
    def __init__(self, segments):
        self.items = segments
        self.size = len(segments)
        self.centroid = self._calculate_centroid()
        self.radius = self._calculate_radius()
        self.merged = False

    def _calculate_centroid(self):
        total_x = 0
        total_y = 0
        for seg in self.items:
            start, end = seg.points[0], seg.points[-1]
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            total_x += mid_x
            total_y += mid_y
        centroid_x = total_x / self.size
        centroid_y = total_y / self.size
        return (centroid_x, centroid_y)

    def _calculate_radius(self):
        max_distance = 0
        for seg in self.items:
            start, end = seg.points[0], seg.points[-1]
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            distance = self.compute_point_distance((mid_x, mid_y), self.centroid)
            if distance > max_distance:
                max_distance = distance
        return max_distance

    @staticmethod
    def compute_point_distance(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_angular_distance(seg1, seg2):
    start1, end1 = seg1.points[0], seg1.points[-1]
    start2, end2 = seg2.points[0], seg2.points[-1]
    vector1 = (end1[0] - start1[0], end1[1] - start1[1])
    vector2 = (end2[0] - start2[0], end2[1] - start2[1])
    angle1 = atan2(vector1[1], vector1[0])
    angle2 = atan2(vector2[1], vector2[0])
    angle_diff = abs(angle1 - angle2)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    len1 = Cluster.compute_point_distance(start1, end1)
    len2 = Cluster.compute_point_distance(start2, end2)
    return abs(sin(angle_diff)) * max(len1, len2)


def compute_vector_distance(r1, r2):
    sum_square = torch.sum((r1 - r2) ** 2)
    return torch.sqrt(sum_square).item()


def calculate_distance(seg1, seg2, alpha, beta, gamma):
    d1 = Cluster.compute_point_distance(
        seg1.points[0], seg2.points[0]
    ) + Cluster.compute_point_distance(seg1.points[-1], seg2.points[-1])
    d2 = compute_angular_distance(seg1, seg2)
    d3 = compute_vector_distance(seg1.emb, seg2.emb)
    return alpha * d1 + beta * d2 + gamma * d3


def secure_set_union(local_cs):
    union_c = []
    for local_c in local_cs:
        union_c.extend(local_c)
    return union_c


def merge_clus(cluster1, cluster2):
    all_segments = cluster1.items + cluster2.items
    return Cluster(all_segments)


def local_clustering(trajectory_segments, eps, min_samples, alpha, beta, gamma):
    num_segments = len(trajectory_segments)
    distance_matrix = np.zeros((num_segments, num_segments))
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            dist = calculate_distance(
                trajectory_segments[i], trajectory_segments[j], alpha, beta, gamma
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(distance_matrix)

    local_clusters = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        cluster_segments = [trajectory_segments[i] for i in cluster_indices]
        cluster = Cluster(cluster_segments)
        local_clusters.append(cluster)

    return local_clusters


def federated_cluster_aggregation(local_cs, eps):
    global_c = secure_set_union(local_cs)
    dset = []
    for i in range(len(global_c)):
        global_c[i].merged = False
        for j in range(i + 1, len(global_c)):
            d_ij = Cluster.compute_point_distance(
                global_c[i].centroid, global_c[j].centroid
            )
            dset.append((global_c[i], global_c[j], d_ij))
    dset.sort(key=lambda x: x[2])
    for c_i, c_j, d_ij in dset:
        if not c_i.merged and not c_j.merged and d_ij < eps:
            c_merged = merge_clus(c_i, c_j)
            c_i.merged = True
            c_j.merged = True
            global_c.append(c_merged)
            global_c.remove(c_i)
            global_c.remove(c_j)
    return global_c


def run_federate_clustering():
    results_path = r"resource\results\test"
    fed_folders = [
        f
        for f in os.listdir(results_path)
        if os.path.isdir(os.path.join(results_path, f)) and f.startswith("fed_")
    ]
    all_info = []
    for i in range(len(fed_folders)):
        fed_path = os.path.join(results_path, f"fed_{i + 1}")
        embs_path = os.path.join(fed_path, "embs.pt")
        segments_path = os.path.join(fed_path, "segments.pkl")
        embs = torch.load(embs_path, weights_only=True)
        segments = pd.read_pickle(segments_path)
        all_info.append({"embs": embs, "segments": segments})

    num_fed = len(all_info)
    fed_segments = []
    for fed_id in range(num_fed):
        traj_data = all_info[fed_id]["segments"]
        segments = []
        for i, row in traj_data.iterrows():
            points = row["merc_seq"]
            if len(points) < 2:
                continue
            segment_id = i
            emb = all_info[fed_id]["embs"][i]
            segment = Segment(segment_id, points, emb)
            segments.append(segment)
        fed_segments.append(segments)

    # 本地聚类参数
    local_eps = 1000.0
    local_min_samples = 2
    alpha = 1
    beta = 1
    gamma = 1

    # 联邦聚类聚合参数
    federated_eps = 3.0

    # 模拟多个参与方的本地聚类
    local_clustering_results = []
    for segments in fed_segments:
        local_clusters = local_clustering(
            segments, local_eps, local_min_samples, alpha, beta, gamma
        )
        local_clustering_results.append(local_clusters)

    # 进行联邦聚类聚合
    global_clusters = federated_cluster_aggregation(
        local_clustering_results, federated_eps
    )

    # 计算轮廓系数
    all_segments = []
    labels = []
    for i, cluster in enumerate(global_clusters):
        for segment in cluster.items:
            all_segments.append(segment)
            labels.append(i)

    num_segments = len(all_segments)
    distance_matrix = np.zeros((num_segments, num_segments))
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            dist = calculate_distance(
                all_segments[i], all_segments[j], alpha, beta, gamma
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
    print(f"Silhouette Coefficient: {silhouette_avg}")

    # 输出结果
    for i, cluster in enumerate(global_clusters):
        print(f"Cluster {i}:")
        print(f"  Centroid: {cluster.centroid}")
        print(f"  Radius: {cluster.radius}")
        print(f"  Size: {cluster.size}")


def run_clustering():
    trajs_filepath = r"./resource/results/t2vec/trajs.pkl"
    embs_filepath = r"./resource/results/t2vec/embs.pkl"

    with open(trajs_filepath, "rb") as f:
        all_trajs = pickle.load(f)

    # 使用 torch 加载嵌入
    all_embeddings = torch.load(embs_filepath)

    all_segments = [
        Segment(i, traj, emb)
        for i, (traj, emb) in enumerate(zip(all_trajs, all_embeddings))
    ]

    # 本地聚类参数
    local_eps = 1000.0
    local_min_samples = 2
    alpha = 1
    beta = 1
    gamma = 1

    # 进行集中式聚类
    local_clusters = local_clustering(
        all_segments, local_eps, local_min_samples, alpha, beta, gamma
    )

    # 计算轮廓系数
    all_seg = []
    labels = []
    for i, cluster in enumerate(local_clusters):
        for segment in cluster.items:
            all_seg.append(segment)
            labels.append(i)

    num_segments = len(all_seg)
    distance_matrix = np.zeros((num_segments, num_segments))
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            dist = calculate_distance(all_seg[i], all_seg[j], alpha, beta, gamma)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
    print(f"Silhouette Coefficient: {silhouette_avg}")

    # 输出结果
    for i, cluster in enumerate(local_clusters):
        print(f"Cluster {i}:")
        print(f"  Centroid: {cluster.centroid}")
        print(f"  Radius: {cluster.radius}")
        print(f"  Size: {cluster.size}")


if __name__ == "__main__":
    run_clustering()
