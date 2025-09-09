import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import find_all_intersections


# Тест для find_all_intersections
def test_find_all_intersections():
    intersections = [
        [0, 1],
        [1, 2],
        [2],
        [3]
    ]
    result = find_all_intersections(0, intersections)
    assert sorted(result) == [0, 1, 2]

    result = find_all_intersections(3, intersections)
    assert result == [3]
