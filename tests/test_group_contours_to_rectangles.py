import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import group_contours_to_rectangles


# Тест для group_contours_to_rectangles
def test_group_contours_to_rectangles():
    contours = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [2, 3, 4, 5]],
        [[9, 10, 11, 12], [3, 8, 7, 10]]
    ]
    expected = [
        [1, 2, 7, 8],
        [3, 8, 11, 12]
    ]
    result = group_contours_to_rectangles(contours)
    assert result == expected

