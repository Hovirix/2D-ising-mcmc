import time

import numpy as np

from .tasks import task_2_parameters, task_3_critical_temp, task_4_clusters


def main():
    start = time.time()
    np.random.seed(42)

    task_2_parameters()
    task_3_critical_temp()
    task_4_clusters()

    print(f"All tasks completed in {time.time() - start:.1f} seconds.")


if __name__ == "__main__":
    main()
