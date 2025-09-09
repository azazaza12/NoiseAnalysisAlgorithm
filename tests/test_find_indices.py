import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import find_indices

def test_find_indices_normal_case():
    maximums = [3, 7, 10]
    minimums = [2, 5, 8]
    result = find_indices(maximums, minimums)
    expected = [(0, 1), (1, 2), (2, -1)]
    assert result == expected

def test_find_indices_empty_minimums():
    maximums = [1, 2]
    minimums = []
    result = find_indices(maximums, minimums)
    expected = [(-1, -1), (-1, -1)]
    assert result == expected
