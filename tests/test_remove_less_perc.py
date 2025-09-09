import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from functions import remove_less_perc_on_origin_panel


def test_remove_less_perc_on_origin_panel():
    # Тест с обычными данными
    aps_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    new_aps_data = np.array([[5, 9, 0], [1, 0, 3], [7, 2, 8]])

    result = remove_less_perc_on_origin_panel(aps_data, new_aps_data)

    expected_result = np.array([[0, 0, 0], [0, 0, 0], [0, 2, 8]])  # обнуляется, что меньше перцентиля
    np.testing.assert_array_equal(result, expected_result)

    # Тест с нулями в новой панели (все значения меньше перцентиля)
    new_aps_data_all_zero = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    result_zero = remove_less_perc_on_origin_panel(aps_data, new_aps_data_all_zero)
    np.testing.assert_array_equal(result_zero, new_aps_data_all_zero)

    # Тест с высокими значениями в aps_data
    aps_data_high = np.array([[85, 9, 36], [41, 77, 6], [23, 90, 19]])
    new_aps_data = np.array([[5, 0, 0], [0, 0, 0], [0, 2, 0]])
    result_high = remove_less_perc_on_origin_panel(aps_data_high, new_aps_data)
    np.testing.assert_array_equal(result_high, new_aps_data)
