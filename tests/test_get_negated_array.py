import pytest
import numpy as np
from scipy.signal import find_peaks
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import get_negated_array

def test_get_negated_array_valid():
    input_data = [[1, -2, 3], [0, 4, -5]]
    result = get_negated_array(input_data)
    expected = [[-1, 2, -3], [0, -4, 5]]
    assert result == expected

def test_get_negated_array_empty():
    result = get_negated_array([])
    assert result == []

def test_get_negated_array_single_empty():
    result = get_negated_array([[]])
    assert result == [[]]
