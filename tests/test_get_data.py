import numpy as np
import lasio
import tempfile
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from unittest import mock
from start_program import get_data

# Тест интеграционный
def test_get_data_success(monkeypatch):
    # Создаём временный файл, но ОТДЕЛЬНО открываем для записи
    with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
        tmp_file_path = tmp_file.name

    # Теперь пишем в файл через lasio (когда файл закрыт)
    las = lasio.LASFile()
    las.set_data(np.array([
        [1000.0, 1.0, 2.0, 3.0],
        [1010.0, 4.0, 5.0, 6.0]
    ]))
    las.write(tmp_file_path)

    # Подменяем функции
    monkeypatch.setattr('start_program.functions.get_count_num_channels', lambda las: 3)
    monkeypatch.setattr('start_program.functions.get_channel_type', lambda las: 'LF')
    monkeypatch.setattr('start_program.initialize_constants', lambda f, d, l, a, c: {'dummy': 1})

    frequencies, depth, aps_data = get_data(tmp_file_path)

    assert len(frequencies) == 3
    assert np.allclose(depth, [1000.0, 1010.0])
    assert aps_data.shape == (2, 3)

    os.unlink(tmp_file_path)  # Удаляем файл после теста

def test_get_data_failure(monkeypatch):
    # Подменяем messagebox на заглушку
    monkeypatch.setattr('start_program.messagebox.showerror', lambda title, msg: None)

    frequencies, depth, aps_data = get_data("non_existing_file.las")

    assert frequencies == []
    assert depth == []
    assert aps_data == []
