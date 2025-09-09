import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions


def test_add_width_intensity_limit(monkeypatch):
    array_of_cnt_ = [
        [0, 0, 1, 1],  # ширина 2
        [0, 0, 2, 2],  # ширина 3
    ]
    max_width_ = 25  # значит, треш по ширине = 5
    aps_data_ = np.full((5, 5), 10)  # вся матрица = 10
    aps_data_[0:3, 0:3] = 20  # повышенная интенсивность для второго контура

    result = functions.add_width_intensity_limit(array_of_cnt_, max_width_, aps_data_)
    assert result == [[0, 0, 2, 2]]  # только второй пройдет (ширина > 0.3 и интенсивность выше медианы 10)

