import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import get_x_coordinates_from_cnt

def test_get_x_coordinates_from_cnt_valid():
    # Массив из двух контуров, каждый с тремя точками
    contours = [
        np.array([[[10, 20]], [[15, 25]], [[20, 30]]]),
        np.array([[[5, 5]], [[10, 10]], [[15, 15]]])
    ]
    result = get_x_coordinates_from_cnt(contours)
    expected = [
        [10, 15, 20],
        [5, 10, 15]
    ]
    assert result == expected

def test_get_x_coordinates_from_cnt_empty():
    with pytest.raises(ValueError):
        get_x_coordinates_from_cnt([])

def test_get_x_coordinates_from_cnt_none():
    with pytest.raises(ValueError):
        get_x_coordinates_from_cnt(None)
