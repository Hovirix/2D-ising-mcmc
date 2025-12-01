import numpy as np


def find_clusters(config):
    """
    Implementation of Algorithm 1: Find Clusters.
    """
    N = config.shape[0]
    mask = np.ones((N, N), dtype=bool)
    cluster_sizes = []

    deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while np.any(mask):
        unvisited_indices = np.argwhere(mask)
        if len(unvisited_indices) == 0:
            break

        idx = np.random.randint(len(unvisited_indices))
        start_node = tuple(unvisited_indices[idx])

        todo = [start_node]
        mask[start_node] = False

        sign = config[start_node]
        count = 0

        while todo:
            curr_r, curr_c = todo.pop()
            count += 1

            for dr, dc in deltas:
                nr, nc = (curr_r + dr) % N, (curr_c + dc) % N

                if mask[nr, nc] and config[nr, nc] == sign:
                    mask[nr, nc] = False
                    todo.append((nr, nc))

        cluster_sizes.append(count)

    return cluster_sizes
