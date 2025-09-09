import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import add_noise_end_limit




# Тест для add_noise_end_limit
def test_add_noise_end_limit(monkeypatch):
    import functions

    class MockConstants:
        TRH_XMAX_MATR = 15

    monkeypatch.setattr(functions.constants_store, "constants", MockConstants)
    array_of_cnt_ = [
        [3, 0, 14, 2],  # xmax < 15 (не должен пройти)
        [5, 1, 15, 4],  # xmax == 15 (должен пройти)
        [6, 2, 16, 5]   # xmax > 15 (должен пройти)
    ]
    result = add_noise_end_limit(array_of_cnt_)
    assert result == [[5, 1, 15, 4], [6, 2, 16, 5]]
