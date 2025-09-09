import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
import cv2
import imutils
from functions import find_contours


def test_find_contours_valid_data():
    # Создаём тестовый массив (например, матрица с прямоугольником)
    test_array = np.array([[0, 0, 0, 0, 0],
                           [0, 255, 255, 255, 0],
                           [0, 255, 255, 255, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)

    contours = find_contours(test_array)

    # Проверяем, что функция вернула хотя бы один контур
    assert len(contours) > 0, "Контуры не найдены"


def test_find_contours_empty_array():
    # Тестируем с пустым массивом
    with pytest.raises(ValueError):
        find_contours(np.array([]))


def test_find_contours_none_array():
    # Тестируем с None
    with pytest.raises(ValueError):
        find_contours(None)


def test_find_contours_no_objects():
    # Тестируем с массивом, не содержащим объектов
    test_array = np.zeros((5, 5), dtype=np.uint8)
    contours, drawing = find_contours(test_array)
    # В этом случае контуров не должно быть
    assert len(contours) == 0, "Контуры найдены, хотя их не должно быть"
