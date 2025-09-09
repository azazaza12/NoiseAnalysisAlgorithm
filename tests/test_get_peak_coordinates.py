import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import get_peak_coordinates

def test_get_peak_coordinates_with_zero_derivative():
    array_of_cnts = [
        np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    ]
    zero_deriv_map = {1: (0, 2)}
    freqs = np.arange(0, 10)
    depth = np.arange(0, 10)

    x, y, y_min = get_peak_coordinates(array_of_cnts, 0, 1, zero_deriv_map, freqs, depth)
    assert x == 5  # max(1,5)
    assert y == 6  # max(2,6)
    assert y_min == 2  # min(2,6)

def test_get_peak_coordinates_without_zero_derivative():
    array_of_cnts = [
        np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    ]
    zero_deriv_map = {}
    freqs = np.arange(0, 10)
    depth = np.arange(0, 10)

    x, y, y_min = get_peak_coordinates(array_of_cnts, 0, 1, zero_deriv_map, freqs, depth)
    assert x == 3
    assert y == 4
    assert y_min == 4



# Тест: координаты пика выходят за частоты или глубины
def test_get_peak_coordinates_overflow():
    array_of_cnts = [
        np.array([[[20, 30]]])  # координаты заведомо за пределами частот и глубин
    ]
    zero_deriv_map = {}
    freqs = np.arange(0, 10)  # длина 10
    depth = np.arange(0, 10)  # длина 10

    x, y, y_min = get_peak_coordinates(array_of_cnts, 0, 0, zero_deriv_map, freqs, depth)
    assert x == 9  # максимальный допустимый индекс
    assert y == 9
    assert y_min == 9
