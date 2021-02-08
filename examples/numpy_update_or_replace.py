from time import time

import numpy as np


def replace(nrows=1000, num=100000):
    n = np.ones((nrows, 2))
    for i in range(num):
        n = n + 1
    return n


def update(nrows=1000, num=100000):
    n = np.ones((nrows, 2))
    for i in range(num):
        n[:] = n[:] + 1

    return n


if __name__ == '__main__':
    t1 = time()
    replace()
    t2 = time()
    update()
    t3 = time()

    print(f"replace: {t2 - t1}; update: {t3 - t2}")
