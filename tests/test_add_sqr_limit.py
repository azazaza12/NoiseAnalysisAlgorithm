import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def test_add_sqr_limit(monkeypatch):
    import functions

    class MockConstants:
        TRH_SQR = 4

    monkeypatch.setattr(functions.constants_store, "constants", MockConstants)
    array_of_cnt_ = [
        [0, 0, 1, 1],  # Площадь 4 (2x2), порог 4 → не пройдет
        [0, 0, 2, 2],  # Площадь 9 (3x3), порог 4 → пройдет
    ]
    result, max_width = functions.add_sqr_limit(array_of_cnt_, frequencies_=None, depth_=None)
    assert result == [[0, 0, 2, 2]]
    assert max_width == 3  # Ширина самого широкого прямоугольника 3


