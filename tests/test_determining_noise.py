import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions

def test_determiningNoise_reservoir_ru():
    points = [
        {'x_min': 5, 'x_max': 6, 'y_min': 1, 'y_max': 1}  # df > dd → Reservoir
    ]
    frequencies = np.array([1, 5, 6, 10, 20])
    depth = np.array([0, 1, 2, 3])
    result = functions.determiningNoise(points, frequencies, depth, language_="r")
    assert result == ["Поток по пласту"]

def test_determiningNoise_chanelling_ru():
    points = [
        {'x_min': 2, 'x_max': 2, 'y_min': 0, 'y_max': 3}  # df < dd → Chanelling
    ]
    frequencies = np.array([1, 2, 3, 4])
    depth = np.array([0, 1, 2, 3])
    result = functions.determiningNoise(points, frequencies, depth, language_="r")
    assert result == ["Заколонная циркуляция"]

def test_determiningNoise_borehole_ru():
    points = [
        {'x_min': 0, 'x_max': 1, 'y_min': 0, 'y_max': 0}  # else → Borehole
    ]
    frequencies = np.array([0, 1, 2, 3])
    depth = np.array([0, 1, 2, 3])
    result = functions.determiningNoise(points, frequencies, depth, language_="r")
    assert result == ["Борхол"]
