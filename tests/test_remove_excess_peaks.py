import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import remove_excess_peaks


def test_remove_excess_peaks_basic_case():
    cnts = [np.array([[[1, 2]], [[3, 4]], [[5, 6]]])]
    maxs = [[1]]
    mins = [[0, 2]]
    x_coords = [[1, 3, 5]]
    zero_deriv = [[{1: (0, 2)}]]
    freqs = np.arange(0, 10)
    depth = np.arange(0, 10)

    result = remove_excess_peaks(cnts, maxs, mins, x_coords, zero_deriv, freqs, depth)
    assert len(result) == 1  # один объект
    assert isinstance(result[0], list)

def test_remove_excess_peaks_no_peaks():
    cnts = [np.array([[[1, 2]], [[3, 4]], [[5, 6]]])]
    maxs = [[]]
    mins = [[]]
    x_coords = [[1, 3, 5]]
    zero_deriv = [[]]
    freqs = np.arange(0, 10)
    depth = np.arange(0, 10)

    result = remove_excess_peaks(cnts, maxs, mins, x_coords, zero_deriv, freqs, depth)
    assert result == [[1, 2, 5, 6]]  # общий прямоугольник


# Тест: если массивы минимумов пустые
def test_remove_excess_peaks_with_empty_mins():
    cnts = [np.array([[[1, 2]], [[3, 4]], [[5, 6]]])]
    maxs = [[1]]
    mins = [[]]  # пустые минимумы
    x_coords = [[1, 3, 5]]
    zero_deriv = [[{1: (0, 2)}]]
    freqs = np.arange(0, 10)
    depth = np.arange(0, 10)

    # не должно упасть, должно объединить пик как отдельный объект
    result = remove_excess_peaks(cnts, maxs, mins, x_coords, zero_deriv, freqs, depth)
    assert isinstance(result, list)

# Тест: все пиковые индексы равны
def test_remove_excess_peaks_all_peaks_same():
    cnts = [np.array([[[1, 2]], [[3, 4]], [[5, 6]]])]
    maxs = [[0, 0, 0]]  # все пики указывают на одно и то же
    mins = [[1, 2]]
    x_coords = [[1, 3, 5]]
    zero_deriv = [[{0: (0, 2)}]]
    freqs = np.arange(0, 10)
    depth = np.arange(0, 10)

    result = remove_excess_peaks(cnts, maxs, mins, x_coords, zero_deriv, freqs, depth)
    assert len(result) > 0  # должно вернуть хотя бы один объект