import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions


def test_wide_contours():

    array_of_cnts = [
        [0, 2, 1, 2],  # ymin=2 ymax=2
        [0, 4, 1, 4],  # ymin=4 ymax=4 (расширится)
    ]
    aps_data_ = np.zeros((6, 6))

    result = functions.wide_contours(array_of_cnts, aps_data_)
    assert result == [
        [0, 1, 1, 3],  # расширен на глубину вверх и вниз
        [0, 3, 1, 5]   # тоже расширен (вниз ограничен depth_)
    ]
