import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import find_contour_indexes_to_keep

def test_find_contour_indexes_to_keep():
    cnts = [
        [0, 0, 2, 2],  # прямоугольник 1
        [5, 5, 7, 7],  # прямоугольник 2 (не пересекается с первым)
    ]
    result = find_contour_indexes_to_keep(cnts)
    assert result == [0, 1], "Оба прямоугольника должны остаться (нет пересечений)"

def find_contour_indexes_to_keep():
    cnts = [
        [0, 0, 4, 4],  # большой прямоугольник
        [2, 2, 3, 3],  # маленький, внутри первого
    ]
    result = find_contour_indexes_to_keep(cnts)
    assert result == [0], "Должен остаться только большой прямоугольник"

def test_find_contour_indexes_to_keep_equal_area():
    cnts = [
        [0, 0, 2, 2],  # площадь 9
        [2, 2, 4, 4],  # площадь 9, пересечение по углу
    ]
    result = find_contour_indexes_to_keep(cnts)
    assert sorted(result) == [0, 1], "Оба должны остаться (равные по площади)"
