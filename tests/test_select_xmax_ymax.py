import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import select_xmax_ymax_from_2_cnt_index



def test_select_xmax_ymax_from_2_cnt_index_basic():
    array_of_cnts = [
        np.array([[[1, 2]], [[5, 6]], [[3, 4]]])
    ]
    xmax, ymax, ymin = select_xmax_ymax_from_2_cnt_index(array_of_cnts, 0, 0, 1)
    assert xmax == 5
    assert ymax == 6
    assert ymin == 2


