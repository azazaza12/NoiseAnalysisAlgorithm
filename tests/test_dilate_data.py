import numpy as np
import pytest
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import dilate_data

def test_dilate_data_none():
    with pytest.raises(ValueError):
        dilate_data(None)

def test_dilate_data_empty():
    with pytest.raises(ValueError):
        dilate_data(np.array([]))

def test_dilate_data_valid_array():
    arr = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]], dtype='uint8')
    result = dilate_data(arr)
    # Проверяем, что форма не изменилась
    assert result.shape == arr.shape
    # Проверяем, что количество единиц увеличилось после дилатации
    assert np.sum(result > 0) > np.sum(arr > 0)
