import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions


def test_calculate_amplitudes_of_signal():
    points = [
        {'x_min': 1, 'x_max': 3, 'y_min': 10, 'y_max': 12}
    ]
    frequencies = np.array([0, 1, 2, 3, 4])
    depth = np.array([9, 10, 11, 12, 13])
    aps_data_ = np.array([
        [1, 2, 3, 4, 5],
        [5, 5, 5, 5, 5],
        [10, 10, 10, 10, 10],
        [20, 20, 20, 20, 20],
        [30, 30, 30, 30, 30]
    ])

    result = functions.calculate_amplitudes_of_signal(points, frequencies, depth, aps_data_)
    assert len(result) == 1
    assert result[0] == 20  # 99-й перцентиль для выбранного диапазона будет 20


