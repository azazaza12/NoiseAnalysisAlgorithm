import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import upload_report_to_excel



def test_upload_report_to_excel(tmp_path):
    # Данные для теста
    points = [
        {'x_min': 1.0, 'x_max': 5.0, 'y_min': 100.0, 'y_max': 120.0}
    ]
    frequencies = np.linspace(0, 10, 11)  # 0, 1, 2, ..., 10
    depth = np.linspace(90, 130, 5)  # 90, 100, 110, 120, 130
    aps_data = np.random.rand(len(depth), len(frequencies)) * 100  # Матрица амплитуд
    file_path = tmp_path / "test_report.xlsx"

    upload_report_to_excel(points, frequencies, depth, aps_data, 'e', str(file_path))

    # Проверка, что файл создан
    assert os.path.exists(file_path)

    # Проверка содержимого
    df = pd.read_excel(file_path, index_col=0)
    assert "Frequency range, kHz " in df.columns
    assert "Amplitude, dB" in df.columns
    assert len(df) == 1
