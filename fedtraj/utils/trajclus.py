import numpy as np


def d_euclidean(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_point_projection_on_line(point, line):
    """
    Calculate the projection of a point onto a line.
    """
    line_vector = line[1] - line[0]
    point_vector = point - line[0]
    dot_product = np.dot(point_vector, line_vector)
    line_length_squared = np.dot(line_vector, line_vector)
    # 检查 line_length_squared 是否为 0
    if line_length_squared == 0:
        # 如果为 0，说明两点重合，投影就是这个重合点
        return line[0]
    projection = line[0] + (dot_product / line_length_squared) * line_vector
    return projection


def d_perpendicular(l1, l2):
    """
    Calculate the perpendicular distance between two lines.
    """
    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    lehmer_1 = d_euclidean(l_shorter[0], ps)
    lehmer_2 = d_euclidean(l_shorter[-1], pe)

    if lehmer_1 == 0 and lehmer_2 == 0:
        return 0
    return (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)


def d_parallel(l1, l2):
    """
    Calculate the parallel distance between two lines.
    """
    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    parallel_1 = min(d_euclidean(l_longer[0], ps), d_euclidean(l_longer[-1], ps))
    parallel_2 = min(d_euclidean(l_longer[0], pe), d_euclidean(l_longer[-1], pe))

    return min(parallel_1, parallel_2)


def d_angular(l1, l2, directional=True):
    """
    Calculate the angular distance between two lines.
    """

    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    # Get the minimum intersecting angle between both lines
    shorter_slope = (
        (l_shorter[-1, 1] - l_shorter[0, 1]) / (l_shorter[-1, 0] - l_shorter[0, 0])
        if l_shorter[-1, 0] - l_shorter[0, 0] != 0
        else np.inf
    )
    longer_slope = (
        (l_longer[-1, 1] - l_longer[0, 1]) / (l_longer[-1, 0] - l_longer[0, 0])
        if l_longer[-1, 0] - l_longer[0, 0] != 0
        else np.inf
    )

    # The case of a vertical line
    theta = None
    if np.isinf(shorter_slope):
        # Get the angle of the longer line with the x-axis and subtract it from 90 degrees
        tan_theta0 = longer_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    elif np.isinf(longer_slope):
        # Get the angle of the shorter line with the x-axis and subtract it from 90 degrees
        tan_theta0 = shorter_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    else:
        tan_theta0 = (shorter_slope - longer_slope) / (1 + shorter_slope * longer_slope)
        tan_theta1 = tan_theta0 * -1

        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))

        theta = min(theta0, theta1)

    if directional:
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])

    if 0 <= theta < (90 * np.pi / 180):
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
    elif (90 * np.pi / 180) <= theta <= np.pi:
        return np.sin(theta)
    else:
        raise ValueError("Theta is not in the range of 0 to 180 degrees.")


# Minimum Description Length
def minimum_desription_length(
    start_idx,
    curr_idx,
    trajectory,
    w_angular=1,
    w_perpendicular=1,
    par=True,
    directional=True,
):
    """
    Calculate the minimum description length.
    """
    LH = LDH = 0
    for i in range(start_idx, curr_idx - 1):
        ed = d_euclidean(trajectory[i], trajectory[i + 1])
        LH += max(0, np.log2(ed, where=ed > 0))
        if par:
            for j in range(start_idx, i - 1):

                _d_perpendicular = d_perpendicular(
                    np.array([trajectory[start_idx], trajectory[i]]),
                    np.array([trajectory[j], trajectory[j + 1]]),
                )
                _d_angular = d_angular(
                    np.array([trajectory[start_idx], trajectory[i]]),
                    np.array([trajectory[j], trajectory[j + 1]]),
                    directional=directional,
                )

                LDH += w_perpendicular * _d_perpendicular
                LDH += w_angular * _d_angular

    if par:
        return LH + LDH
    return LH


def traclus_partition(
    trajectory, directional=True, progress_bar=False, w_perpendicular=1, w_angular=1
):
    """
    Partition a trajectory into segments.
    """

    # Ensure that the trajectory is a numpy array of shape (n, 2)
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)
    if trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be a numpy array of shape (n, 2)")
    
    traj_len = trajectory.shape[0]
    if traj_len == 0:
        return np.array([]), np.zeros(0, dtype=bool)

    # Initialize the characteristic points, add the first point as a characteristic point
    cp_indices = []
    cp_indices.append(0)

    traj_len = trajectory.shape[0]
    start_idx = 0

    length = 1
    while start_idx + length < traj_len:
        if progress_bar:
            print(f"\r{round(((start_idx + length) / traj_len) * 100, 2)}%", end="")
        # print(f'Current Index: {start_idx + length}, Trajectory Length: {traj_len}')
        curr_idx = start_idx + length
        # print(start_idx, curr_idx)
        # print(f"Current Index: {curr_idx}, Current point: {trajectory[curr_idx]}")
        cost_par = minimum_desription_length(
            start_idx,
            curr_idx,
            trajectory,
            w_angular=w_angular,
            w_perpendicular=w_perpendicular,
            directional=directional,
        )
        cost_nopar = minimum_desription_length(
            start_idx, curr_idx, trajectory, par=False, directional=directional
        )
        # print(f'Cost with partition: {cost_par}, Cost without partition: {cost_nopar}')
        if cost_par > cost_nopar:
            # print(f"Added characteristic point: {trajectory[curr_idx-1]} with index {curr_idx-1}")
            cp_indices.append(curr_idx - 1)
            start_idx = curr_idx - 1
            length = 1
        else:
            length += 1

    # Add last point to characteristic points
    cp_indices.append(len(trajectory) - 1)
    # print(cp_indices)

    # Create the mask array
    mask = np.zeros(traj_len, dtype=bool)
    mask[cp_indices] = True

    return np.array([trajectory[i] for i in cp_indices]), mask