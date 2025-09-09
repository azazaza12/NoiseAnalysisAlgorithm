import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import get_count_num_channels  # Замени 'your_module' на правильное имя файла

class DummyLas:
    def __init__(self, curvesdict):
        self.curvesdict = curvesdict

def test_get_count_num_channels_aps():
    las = DummyLas({
        'APS[1]': None,
        'APS[2]': None,
        'APS[3]': None
    })
    assert get_count_num_channels(las) == 3

def test_get_count_num_channels_aps_lf():
    las = DummyLas({
        'APS_LF[1]': None,
        'APS_LF[2]': None
    })
    assert get_count_num_channels(las) == 2

def test_get_count_num_channels_snl_flowing():
    las = DummyLas({
        'SNL_FlOWING[1]': None,
        'SNL_FlOWING[2]': None,
        'SNL_FlOWING[3]': None,
        'SNL_FlOWING[4]': None
    })
    assert get_count_num_channels(las) == 4

def test_get_count_num_channels_no_channels():
    las = DummyLas({})
    assert get_count_num_channels(las) == 0
