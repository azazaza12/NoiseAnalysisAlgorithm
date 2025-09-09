import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import select_common_rectangle



def test_select_common_rectangle_basic():
    cnts = [
        [1, 2, 5, 6],
        [0, 3, 4, 7],
        [2, 1, 6, 5]
    ]
    result = select_common_rectangle(cnts)
    assert result == [0, 1, 6, 7]  # объединённый прямоугольник


# Тест: пустые входные контуры
def test_select_common_rectangle_empty_input():
    with pytest.raises(IndexError):
        select_common_rectangle([])  # ожидаем ошибку, т.к. вход пустой
