import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import delete_small_contours


def test_delete_small_contours():
    cnts = [
        [0, 0, 1, 1],  # площадь 4, меньше min_square=5
        [0, 0, 2, 2],  # площадь 9, меньше min_square=10
        [0, 0, 3, 3],  # площадь 16, больше min_square=10
    ]
    aps_data = np.full((5, 5), 5)  # Основной массив с амплитудой 5
    aps_data[0:3, 0:3] = 10  # Прямоугольник (0,0,2,2) будет иметь среднее значение 10
    aps_data[3:5, 3:5] = 2   # Прямоугольник (0,0,3,3) будет иметь среднее значение 2

    result = delete_small_contours(cnts, aps_data, min_square=10)
    assert result == [[0, 0, 3, 3]]  # Останется только прямоугольник с площадью 16, так как среднее значение 2 меньше, чем для других



