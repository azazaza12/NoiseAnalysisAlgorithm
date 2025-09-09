import numpy as np
import pytest
import sys
import os
from scipy.signal import medfilt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants_store
import functions  # для find_contours
from start_program import detect_objects

def test_detect_objects_returns_rects(monkeypatch):
    # Настраиваем фиктивные константы
    class DummyConstants:
        MEDFILT_WINDOW_ONE_1 = 25
        MEDFILT_WINDOW_ONE_2 = 1
        MEDFILT_WINDOW_TWO_1 = 1
        MEDFILT_WINDOW_TWO_2 = 9
        TRH_SQR = 50
        TRH_XMIN_MATR = 40
        TRH_XMAX_MATR = 15
        THIKNES = 1
    constants_store.constants = DummyConstants()

    # Генерируем стабильные тестовые данные
    frequencies = np.linspace(0.1, 10, 50)
    depth = np.linspace(1000, 2000, 100)

    aps_data = np.random.normal(2, 0.5, (100, 50))  # равномерный слабый шум со средней амплитудой 2

    # Добавляем прямоугольный сигнальный объект с высокой контрастностью
    aps_data[20:40, 50:75] = 50  # мощный прямоугольный сигнал
    print(f"aps:{aps_data}")

    # Отладка: проверяем контуры до вызова detect_objects
    count_new_up_lines = round(constants_store.constants.MEDFILT_WINDOW_ONE_1 / 4)
    new_lines_aps_data = functions.add_new_lines(aps_data, count_new_up_lines)
    medfilt_aps_data = medfilt(new_lines_aps_data, [constants_store.constants.MEDFILT_WINDOW_ONE_1,
                                                    constants_store.constants.MEDFILT_WINDOW_ONE_2])
    print(f"medfilt1:{medfilt_aps_data}")
    sub_lines_aps_data = new_lines_aps_data - medfilt_aps_data
    med_sub_lines_aps_data = medfilt(sub_lines_aps_data, [constants_store.constants.MEDFILT_WINDOW_TWO_1,
                                                          constants_store.constants.MEDFILT_WINDOW_TWO_2])
    print(f"medfilt2:{med_sub_lines_aps_data}")
    median_drift = med_sub_lines_aps_data[count_new_up_lines:-count_new_up_lines, :]
    print(f"median drift:{median_drift}")

    blured_data = functions.blur_data(median_drift)
    print(f"blut:{blured_data}")
    binarised_aps_data = functions.binarisation(aps_data, blured_data)
    print(f"bin:{binarised_aps_data}")

    dilated_data = functions.dilate_data(binarised_aps_data)
    print(f"dilate:{dilated_data}")

    array_of_contours = functions.find_contours(dilated_data)
    print(f"arr cnts:{array_of_contours}")
    print(f" Найдено контуров: {len(array_of_contours)}")

    assert array_of_contours is not None, "Контуры не найдены (None)"
    assert len(array_of_contours) > 0, "Контуры не найдены (пусто). Проверь параметры"

    # Вызываем функцию
    rects = detect_objects(frequencies, depth, aps_data)

    # Проверки
    assert rects is not None, "Функция должна возвращать список прямоугольников"
    assert isinstance(rects, list), "Результат должен быть списком"
    assert all(isinstance(r, dict) or isinstance(r, tuple) or isinstance(r, list) for r in rects), "Каждый прямоугольник должен быть структурой данных"
