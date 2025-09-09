import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import add_noise_start_limit


def test_add_noise_start_limit(monkeypatch):
    import functions

    class MockConstants:
        TRH_XMIN_MATR = 5

    monkeypatch.setattr(functions.constants_store, "constants", MockConstants)

    array_of_cnt_ = [
        [3, 0, 6, 2],
        [5, 1, 8, 4],
        [6, 2, 9, 5]
    ]
    result = add_noise_start_limit(array_of_cnt_)
    assert result == [[3, 0, 6, 2]]


