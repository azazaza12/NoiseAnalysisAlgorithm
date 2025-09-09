import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import merge_contours



# Тест для merge_contours
def test_merge_contours():
    array_of_cnt_ = np.array([
        [1, 1, 3, 3],
        [2, 2, 4, 4],
        [10, 10, 12, 12]
    ])
    all_intersections_list_ = [
        [0, 1],  # 0 пересекается с 1
        [1, 0],  # 1 пересекается с 0
        [2]      # 2 не пересекается ни с кем
    ]
    result = merge_contours(array_of_cnt_, all_intersections_list_)
    assert result == [[1, 1, 4, 4], [10, 10, 12, 12]]  # Первый и второй объединяются, третий отдельно

