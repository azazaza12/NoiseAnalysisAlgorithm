import numpy as np
import pytest
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import remove_background_noise, blur_data, binarisation

# Для binarisation нужна заглушка remove_less_86perc_on_origin_panel
# Предположим, он просто возвращает свой второй аргумент без изменений
import functions



def test_remove_background_noise_none():
    result = remove_background_noise(None)
    assert result is None

def test_remove_background_noise_empty():
    arr = np.array([])
    result = remove_background_noise(arr)
    np.testing.assert_array_equal(result, arr)

def test_remove_background_noise_simple_case():
    arr = np.array([[1, 1], [1, 1]])
    result = remove_background_noise(arr)
    assert result.shape == arr.shape
    # После обработки там не должно остаться единиц в исходной форме

def test_blur_data_none():
    with pytest.raises(ValueError):
        blur_data(None)

def test_blur_data_empty():
    with pytest.raises(ValueError):
        blur_data(np.array([]))

def test_blur_data_valid_array():
    arr = np.random.rand(10, 10).astype(np.float32)
    result = blur_data(arr)
    assert result.shape == arr.shape
    assert result.dtype == np.float32

def test_binarisation_none():
    with pytest.raises(ValueError):
        binarisation(np.random.rand(10, 10), None)

def test_binarisation_empty():
    with pytest.raises(ValueError):
        binarisation(np.random.rand(10, 10), np.array([]))

def test_binarisation_valid():
    aps_data = np.random.rand(10, 10) * 100
    new_aps_data = np.ones((10, 10)) * 5
    result = binarisation(aps_data, new_aps_data)
    assert result.shape == new_aps_data.shape
    # После binarisation значения должны быть только 0 или 1
    unique_vals = np.unique(result)
    assert set(unique_vals).issubset({0, 1})
