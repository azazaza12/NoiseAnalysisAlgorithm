import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import createRects


def test_createRects():
    points = [
        [0, 0, 1, 1],  # (x_min_idx, y_min_idx, x_max_idx, y_max_idx)
        [1, 1, 2, 2]
    ]
    frequencies = np.array([1.0, 2.0, 3.0])
    depth = np.array([100.0, 110.0, 120.0])
    rects = createRects(points, frequencies, depth)

    assert rects == [
        [1.0, 100.0, 2.0, 110.0],
        [2.0, 110.0, 3.0, 120.0]
    ]
