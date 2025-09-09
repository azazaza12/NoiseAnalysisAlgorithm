import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from functions import add_new_lines


def test_add_new_lines():
    # Тест с обычным массивом
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = add_new_lines(array, 2)

    expected_result = np.array([[4, 5, 6], [1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9], [4, 5, 6]])
    np.testing.assert_array_equal(result, expected_result)

    # Тест с пустым массивом (должен выбросить ValueError)
    with pytest.raises(ValueError):
        add_new_lines(np.array([]), 2)

    # Тест с массивом, где количество новых строк = 0
    result_zero = add_new_lines(array, 0)
    np.testing.assert_array_equal(result_zero, array)

    # Тест с маленьким массивом (1x3)
    small_array = np.array([[1, 2, 3]])
    result_small = add_new_lines(small_array, 1)
    expected_small_result = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(result_small, expected_small_result)
