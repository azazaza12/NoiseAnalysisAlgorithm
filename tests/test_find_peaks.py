import pytest
import numpy as np
from scipy.signal import find_peaks
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import find_peaks_function, get_negated_array

def test_find_peaks_function_valid():
    # Входные данные: два простых контура с пиками в центре
    input_data = [[1, 3, 1], [2, 5, 2]]
    result = find_peaks_function(input_data)
    expected = [np.array([1]), np.array([1])]
    # Сравниваем массивы пиков поэлементно
    for res, exp in zip(result, expected):
        np.testing.assert_array_equal(res, exp)

def test_find_peaks_function_empty():
    with pytest.raises(ValueError):
        find_peaks_function([])

def test_find_peaks_function_none():
    with pytest.raises(ValueError):
        find_peaks_function(None)


