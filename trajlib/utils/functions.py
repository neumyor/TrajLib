import numpy as np
import random
import math

from trajlib.utils import tool_funcs
from trajlib.utils.rdp import rdp
from trajlib.utils.cellspace import CellSpace
from trajlib.utils.tool_funcs import truncated_rand


def straight(src):
    """
    This function returns the input trajectory without any modification.

    Args:
        src (list): A list of points representing the input trajectory.

    Returns:
        list: The input trajectory.
    """
    return src


def simplify(src, epsilon):
    """
    This function simplifies the input trajectory using the Ramer-Douglas-Peucker algorithm.

    Args:
        src (list): A list of points representing the input trajectory.
        epsilon (float): The maximum distance between the original curve and its approximation.

    Returns:
        list: The simplified trajectory.
    """
    # src: [[lon, lat], [lon, lat], ...]
    return rdp(src, epsilon=epsilon)


def shift(src, mu=0, sigma=0.5, factor=100, bound_lo=-100, bound_hi=100):
    """
    This function shifts each point in the input trajectory by a random amount.

    Args:
        src (list): A list of points representing the input trajectory.
        mu (float): The mean of the normal distribution used to generate the random shift.
        sigma (float): The standard deviation of the normal distribution used to generate the random shift.
        factor (int): A factor used to scale the random shift.
        bound_lo (int): The lower bound of the random shift.
        bound_hi (int): The upper bound of the random shift.

    Returns:
        list: The shifted trajectory.
    """
    return [[p[0] + truncated_rand(mu, sigma, factor, bound_lo, bound_hi), p[1] + truncated_rand(mu, sigma, factor, bound_lo, bound_hi)] for p in src]


def mask(src, mask_ratio):
    """
    This function masks a random subset of points in the input trajectory.

    Args:
        src (list): A list of points representing the input trajectory.
        mask_ratio (float): The ratio of points to be masked.

    Returns:
        list: The masked trajectory.
    """
    l = len(src)
    arr = np.array(src)
    mask_idx = np.random.choice(l, int(l * mask_ratio), replace=False)
    return np.delete(arr, mask_idx, 0).tolist()


def subset(src, subset_ratio):
    """
    This function returns a random subset of points from the input trajectory.

    Args:
        src (list): A list of points representing the input trajectory.
        subset_ratio (float): The ratio of points to be included in the subset.

    Returns:
        list: The subset of the input trajectory.
    """
    l = len(src)
    max_start_idx = l - int(l * subset_ratio)
    start_idx = random.randint(0, max_start_idx)
    end_idx = start_idx + int(l * subset_ratio)
    return src[start_idx: end_idx]


def translate(src, cell_size):
    """
    This function translates the input trajectory by a random distance and angle.

    Args:
        src (list): A list of points representing the input trajectory.
        cell_size (float): The size of the cell used to generate the random translation.

    Returns:
        list: The translated trajectory.
    """
    rand = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(0, cell_size)
    x = float(np.cos(rand) * dist)
    y = float(np.sin(rand) * dist)
    return [[p[0] + x, p[1] + y] for p in src]

def time_shift(src, factor=3/4):
    """
    This function shifts each point in the input trajectory by a weighted average of its current position and the next position.

    Args:
        src (list): A list of points representing the input trajectory.
        factor (float): The weight of the current position in the weighted average.

    Returns:
        list: The time-shifted trajectory.
    """
    return [
        [
            src[idx][0] * factor + src[idx + 1][0] * (1 - factor),
            src[idx][1] * factor + src[idx + 1][1] * (1 - factor),
        ]
        for idx in range(len(src) - 1)
    ]


def scaling(src, scaling_rate=None):
    """
    This function scales the input trajectory by a random factor.

    Args:
        src (list): A list of points representing the input trajectory.
        scaling_rate (float): The scaling factor. If None, a random factor between 0.7 and 1 will be used.

    Returns:
        list: The scaled trajectory.
    """
    l = len(src)
    x = src[0][0] * 1 / 2 + src[l - 1][0] * 1 / 2
    y = src[0][1] * 1 / 2 + src[l - 1][1] * 1 / 2
    if scaling_rate is None:
        scaling_rate = np.random.uniform(0.7, 1)
    return [
        [
            p[0] * scaling_rate + x * (1 - scaling_rate),
            p[1] * scaling_rate + y * (1 - scaling_rate),
        ]
        for p in src
    ]


def down_sampling(src, threshold=0.7):
    """
    This function downsamples the input trajectory by removing points that are not turning points.

    Args:
        src (list): A list of points representing the input trajectory.
        threshold (float): The probability threshold for keeping a point.

    Returns:
        list: The downsampled trajectory.
    """
    def get_angle(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return np.pi
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        return angle

    l = len(src)
    new_src = []
    for i in range(l):
        if i != 0 and i != l - 1:
            v1 = [src[i][0] - src[i - 1][0], src[i][1] - src[i - 1][1]]
            v2 = [src[i][0] - src[i + 1][0], src[i][1] - src[i + 1][1]]
            if -1 / 2 * np.pi <= get_angle(v1, v2) <= 1 / 2 * np.pi:
                continue
        if np.random.rand() < threshold:
            new_src.append(src[i])
    return new_src


def splicing(src, head_ratio=0.1, tail_ratio=0.1):
    """
    This function splices the input trajectory by adding random points at the beginning and end.

    Args:
        src (list): A list of points representing the input trajectory.
        head_ratio (float): The ratio of points to be added at the beginning.
        tail_ratio (float): The ratio of points to be added at the end.

    Returns:
        list: The spliced trajectory.
    """
    l = len(src)
    new_src_head = []
    for i in range(int(l * head_ratio)):
        if not new_src_head:
            new_src_head.append(
                [src[0][0] + truncated_rand(), src[0][1] + truncated_rand()]
            )
        else:
            new_src_head.append(
                [
                    new_src_head[i - 1][0] + truncated_rand(),
                    new_src_head[i - 1][1] + truncated_rand(),
                ]
            )
    new_src_head.reverse()
    new_src_tail = []
    for i in range(int(l * tail_ratio)):
        if not new_src_tail:
            new_src_tail.append(
                [src[l - 1][0] + truncated_rand(), src[l - 1][1] + truncated_rand()]
            )
        else:
            new_src_tail.append(
                [
                    new_src_tail[i - 1][0] + truncated_rand(),
                    new_src_tail[i - 1][1] + truncated_rand(),
                ]
            )
    return new_src_head + src + new_src_tail


def get_aug_fn(name: str):
    """
    This function returns the augmentation function corresponding to the given name.

    Args:
        name (str): The name of the augmentation function.

    Returns:
        function: The augmentation function.
    """
    return {
        "straight": straight,
        "simplify": simplify,
        "shift": shift,
        "mask": mask,
        "subset": subset,
        "translate": translate,
        "time_shift": time_shift,
        "scaling": scaling,
        "down_sampling": down_sampling,
        "splicing": splicing,
    }.get(name, None)


# pair-wise conversion -- structural features and spatial feasures
def merc2cell2(src, cs: CellSpace):
    """
    This function converts the input trajectory from Mercator coordinates to cell IDs and removes consecutive duplicates.

    Args:
        src (list): A list of points representing the input trajectory.
        cs (CellSpace): The cell space object.

    Returns:
        tuple: A tuple containing the cell IDs and the corresponding points.
    """
    # convert and remove consecutive duplicates
    tgt = [(cs.get_cellid_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i - 1][0]]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p


def generate_spatial_features(src, cs: CellSpace, trajcl_local_mask_sidelen):
    """
    This function generates spatial features for the input trajectory.

    Args:
        src (list): A list of points representing the input trajectory.
        cs (CellSpace): The cell space object.

    Returns:
        list: A list of spatial features for each point in the trajectory.
    """
    # src = [length, 2]
    tgt = []
    lens = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist = (lens[i - 1] + lens[i]) / 2
        dist = dist / (trajcl_local_mask_sidelen / 1.414)  # float_ceil(sqrt(2))

        radian = (
            math.pi
            - math.atan2(src[i - 1][0] - src[i][0], src[i - 1][1] - src[i][1])
            + math.atan2(src[i + 1][0] - src[i][0], src[i + 1][1] - src[i][1])
        )
        radian = 1 - abs(radian) / math.pi

        x = (src[i][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[i][1] - cs.y_min) / (cs.y_max - cs.y_min)
        tgt.append([x, y, dist, radian])

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min) / (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0])

    x = (src[-1][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[-1][1] - cs.y_min) / (cs.y_max - cs.y_min)
    tgt.append([x, y, 0.0, 0.0])
    # tgt = [length, 4]
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length
