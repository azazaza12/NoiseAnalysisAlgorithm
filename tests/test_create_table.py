import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import createTable

def test_createTable():
    rects = [
        {'x_min': 1.0, 'x_max': 5.0, 'y_min': 100.0, 'y_max': 120.0},
        {'x_min': 2.5, 'x_max': 6.0, 'y_min': 150.0, 'y_max': 170.0}
    ]
    object_type = ["Reservoir", "Chanelling"]
    arr_perc99 = [55.6, 60.2]
    table = createTable(rects, object_type, arr_perc99)

    assert table == [
        [100, 120, '1.0-5.0', 56, 'Reservoir'],
        [150, 170, '2.5-6.0', 60, 'Chanelling']
    ]

