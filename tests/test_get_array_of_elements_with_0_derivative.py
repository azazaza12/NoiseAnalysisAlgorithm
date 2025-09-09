import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import get_array_of_elements_with_0_derivative



def test_get_array_of_elements_with_0_derivative_plateau_found():
    peaks = [[2]]
    x_coords = [[0, 1, 1, 2, 3]]  # Производная будет [1, 0, 1, 1], плато на индексах 1-2
    result = get_array_of_elements_with_0_derivative(peaks, x_coords)
    plateau = result[0][0]
    assert 2 in plateau
    start, end = plateau[2]
    assert start == 1 and end == 2

def test_get_array_of_elements_with_0_derivative_no_plateau():
    peaks = [[0]]
    x_coords = [[0, 1, 2, 3]]  # Производная [1,1,1], плато нет
    result = get_array_of_elements_with_0_derivative(peaks, x_coords)
    # Должно быть пусто, так как нет участков с производной 0
    assert result == [[{}]]
