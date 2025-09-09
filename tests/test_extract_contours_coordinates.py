import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import extract_contours_coordinates

# Тесты для draw_cnts_without_x_y_intersections

def test_extract_contours_coordinates_valid_indices():
    cnts = np.array([
        [1, 10, 2, 20],
        [3, 30, 4, 40],
        [5, 50, 6, 60],
    ])
    indices = [0, 2]
    result = extract_contours_coordinates(cnts, indices)
    expected = np.array([
        [1, 10, 2, 20],
        [5, 50, 6, 60],
    ])
    assert np.array_equal(result, expected)

def test_extract_contours_coordinates_invalid_index():
    cnts = np.array([
        [1, 10, 2, 20],
    ])
    with pytest.raises(IndexError):
        extract_contours_coordinates(cnts, [5])

def test_extract_contours_coordinates_empty_array():
    cnts = np.array([])
    with pytest.raises(ValueError):
        extract_contours_coordinates(cnts, [0])


