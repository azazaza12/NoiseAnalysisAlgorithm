import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import find_y_intersections

# Тесты для find_y_intersections
def test_find_y_intersections_basic():
    cnts = np.array([
        [0, 10, 1, 20],  # Прямоугольник 0 пересекается с 2
        [2, 30, 3, 40],  # Прямоугольник 1 ни с кем не пересекается
        [4, 15, 5, 25],  # Прямоугольник 2 пересекается с 0
    ])
    result = find_y_intersections(cnts)
    expected = [
        [0, 2],
        [1],
        [2, 0],
    ]
    assert result == expected

def test_find_y_intersections_empty_array():
    cnts = np.array([])
    with pytest.raises(ValueError):
        find_y_intersections(cnts)
