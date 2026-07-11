from collections import deque

import numpy as np
from scipy.spatial import cKDTree


def flood_indices(xyz, seed, max_distance, max_points, allowed=None):
    """Flood a spatially connected component from seed, capped at max_points."""
    xyz = np.asarray(xyz)
    n = len(xyz)
    if not 0 <= seed < n or max_distance <= 0 or max_points <= 0:
        return np.empty(0, dtype=np.int64)
    allowed = np.ones(n, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    if not allowed[seed]:
        return np.empty(0, dtype=np.int64)

    tree = cKDTree(xyz)
    seen = np.zeros(n, dtype=bool)
    seen[seed] = True
    queue = deque([seed])
    result = []
    while queue and len(result) < max_points:
        index = queue.popleft()
        result.append(index)
        neighbours = tree.query_ball_point(xyz[index], max_distance)
        for neighbour in neighbours:
            if len(result) + len(queue) >= max_points:
                break
            if allowed[neighbour] and not seen[neighbour]:
                seen[neighbour] = True
                queue.append(neighbour)
    return np.asarray(result, dtype=np.int64)


if __name__ == "__main__":
    points = np.array([[0, 0, 0], [0.4, 0, 0], [0.8, 0, 0], [3, 0, 0]])
    assert flood_indices(points, 0, 0.5, 10).tolist() == [0, 1, 2]
    assert flood_indices(points, 0, 0.5, 2).tolist() == [0, 1]
