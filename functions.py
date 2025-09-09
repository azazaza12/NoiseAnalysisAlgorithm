import numpy as np
import cv2
import imutils
from scipy.signal import find_peaks
import pandas as pd
import lasio
import constants_store


# получение количества каналов
def get_count_num_channels(las_):
    num_of_channels = 1
    temp_str = 'APS[' + str(num_of_channels) + ']'
    while temp_str in las_.curvesdict:
        num_of_channels += 1
        temp_str = 'APS[' + str(num_of_channels) + ']'
    if (num_of_channels == 1):
        temp_str = 'APS_LF[' + str(num_of_channels) + ']'
        while temp_str in las_.curvesdict:
            num_of_channels += 1
            temp_str = 'APS_LF[' + str(num_of_channels) + ']'
    if (num_of_channels == 1):
        temp_str = 'SNL_FlOWING[' + str(num_of_channels) + ']'
        while temp_str in las_.curvesdict:
            num_of_channels += 1
            temp_str = 'SNL_FlOWING[' + str(num_of_channels) + ']'
    num_of_channels -= 1
    if num_of_channels == 0:
        raise ValueError("No valid channels found in the data.")
    return num_of_channels


# получение типа канала
def get_channel_type(las_):
    dictionary = las_.curvesdict
    if any(key for key in dictionary.keys() if 'LF' in key):
        return 'LF'
    else:
        return 'HF'


# добавление строк перед применением медфильтра
def add_new_lines(array, count_new_up_or_down_lines):
    if array is None or array.size == 0:
        raise ValueError('Массив пуст, объекты не определены')
    if count_new_up_or_down_lines == 0:
        return array  # Возвращаем массив без изменений
    new_up_lines = []
    for i in range(count_new_up_or_down_lines - 1, -1, -1):
        new_up_lines.append(array[i, :])
    new_down_lines = []
    for i in range(1, count_new_up_or_down_lines + 1):
        new_down_lines.append(array[-i, :])
    new_lines_aps_data = np.vstack([new_up_lines, array, new_down_lines])
    return new_lines_aps_data


# удаляем значения которые меньше 86 персентиля исходной панели
def remove_less_perc_on_origin_panel(aps_data_, new_aps_data_):
    perc = np.percentile(aps_data_, 86)
    temp_aps = np.copy(aps_data_)
    temp_aps[temp_aps <= perc] = 0
    temp_aps[temp_aps > perc] = 1
    new_aps_data_[temp_aps == 0] = 0
    return new_aps_data_


# применяется сужение для удаления ряби
def remove_background_noise(array):
    if array is None or array.size == 0:
        return array
    temp_new_aps_data = array.copy()
    temp_new_aps_data[temp_new_aps_data == 1] = 255
    img = temp_new_aps_data.astype(np.uint8)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    kernel = np.ones((1, 7), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    new_array = np.mean(erosion, axis=2)
    new_array[new_array == 255] = 1
    new_array = np.array(new_array)
    return new_array


# блюр для сглаживания
def blur_data(array):
    if array is not None and array.size != 0:
        img = array.astype(np.float32)
        blured_data = cv2.GaussianBlur(img, (9, 1), 0)
        return blured_data
    else:
        raise ValueError('Массив пуст, объекты не определены')


# приведение к бинарному виду
def binarisation(aps_data_, new_aps_data_):
    if new_aps_data_ is not None and new_aps_data_.size != 0:
        new_aps_data_ = remove_less_perc_on_origin_panel(aps_data_, new_aps_data_)
        new_aps_data_[new_aps_data_ < 4] = 0
        new_aps_data_[new_aps_data_ > 0] = 1
        new_aps_data_ = remove_background_noise(new_aps_data_)
        return new_aps_data_
    else:
        raise ValueError('Массив пуст, объекты не определены')


# расширение по частоте
def dilate_data(array_):
    if array_ is not None and array_.size != 0:
        kernel_dilate = np.ones((1, 3), 'uint8')
        dilated_data = cv2.dilate(array_, kernel_dilate, iterations=4)
        return dilated_data
    else:
        raise ValueError('Массив пуст, объекты не определены')


# нахождение контуров (каждый контур - массив точек)
def find_contours(array):
    if array is not None and array.size != 0:
        dilate_img_uint8 = array.astype(np.uint8)
        contours = cv2.findContours(dilate_img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours2 = imutils.grab_contours(contours)
        return contours2
    else:
        raise ValueError('Массив пуст, объекты не определены')


# получение координат x контуров
def get_x_coordinates_from_cnt(array_of_cnts):
    if array_of_cnts is not None and len(array_of_cnts) != 0:
        x_contours = []
        for contour in array_of_cnts:
            temp_x_contour = []
            for j in range(len(contour)):
                temp_x_contour.append(contour[j][0][0])
            x_contours.append(temp_x_contour)
        return x_contours
    else:
        raise ValueError('Массив пуст, объекты не определены')


# нахождение пиков в каждом контуре
def find_peaks_function(array_of_values):
    if array_of_values is not None and len(array_of_values) != 0:
        new_array_of_values = []
        for cnt in array_of_values:
            new_cnt = cnt.copy()
            # случае если с крайней точки функция уменьшается, то ее нужно считать пиком, втроенная функция find_peaks ее пиком не считаем, поэтому по краям добавляем значения меньше
            new_cnt.insert(0, cnt[0] - 10)
            new_cnt.append(cnt[-1] - 10)
            new_array_of_values.append(new_cnt)
        peaks = []
        for contour in new_array_of_values:
            temp_peak = find_peaks(contour)
            temp_peak = temp_peak[0] - 1
            peaks.append(temp_peak)
        return peaks
    else:
        raise ValueError('Массив пуст, объекты не определены')


# получение обратных х координат (так используя find_peak найдем минимумы)
def get_negated_array(array):
    negated_array = []
    for i in range(len(array)):
        temp_cnt = array[i]
        negated_temp_cnt = list(map(lambda x: -x, temp_cnt))
        negated_array.append(negated_temp_cnt)
    return negated_array


# нахождение индексов минимумов рядом для каждого пика
def find_indices(maximums, minimums):
    indices = []
    minimums = np.asarray(minimums)
    if minimums.size == 0:
        return [(-1, -1)] * len(maximums)
    for element in maximums:
        right_indices = np.where(element < minimums)[0]
        if right_indices.size > 0:
            right = right_indices[0]
            left = right - 1 if right > 0 else -1
        else:
            left = len(minimums) - 1
            right = -1
        indices.append((left, right))
    return indices


# получение xmax, ymax, ymin для участка с 0-ой производной
def select_xmax_ymax_from_2_cnt_index(array_of_cnts_, i, index1, index2):
    x1 = array_of_cnts_[i][index1][0][0]
    y1 = array_of_cnts_[i][index1][0][1]
    x2 = array_of_cnts_[i][index2][0][0]
    y2 = array_of_cnts_[i][index2][0][1]
    xmax = max(x1, x2)
    ymax = max(y1, y2)
    ymin = min(y1, y2)
    return xmax, ymax, ymin


# нахождение участков с 0 производной
def get_array_of_elements_with_0_derivative(array_of_maximums_of_cnt, array_of_x_coordinates_of_cnts):
    array_of_result = []
    for cnt_index, (peaks, x_coords) in enumerate(zip(array_of_maximums_of_cnt, array_of_x_coordinates_of_cnts)):
        derivatives = np.diff(x_coords)
        zero_plateaus = {}
        for peak_idx in peaks:
            start = peak_idx
            end = peak_idx
            # Ищем влево, пока производная = 0
            while start > 0 and derivatives[start - 1] == 0:
                start -= 1
            # Ищем вправо, пока производная = 0
            while end < len(derivatives) and derivatives[end] == 0:
                end += 1
            # Сохраняем, если найдено плато, не совпадающее с одной точкой
            if start != peak_idx or end != peak_idx:
                zero_plateaus[peak_idx] = (start, end)
        array_of_result.append([zero_plateaus])
    return array_of_result


def get_peak_coordinates(array_of_cnts, i, peak, zero_deriv_map, frequencies_, depth_):
    if peak in zero_deriv_map:
        ind1, ind2 = zero_deriv_map[peak]
        return select_xmax_ymax_from_2_cnt_index(array_of_cnts, i, ind1, ind2)
    else:
        x = min(array_of_cnts[i][peak][0][0], len(frequencies_) - 1)
        y = min(array_of_cnts[i][peak][0][1], len(depth_) - 1)
        return x, y, y



# удаление лишних пиков (если пик больше на 1/4 своей ширины (по частоте) чем расстояние по частоте до минимумов справа и слева от него,
# то он выделяется в отдельный объект, иначе объединяется с соседними пиками
def remove_excess_peaks(array_of_cnts, array_of_maximums, array_of_minimums, array_of_x_coordinates,
                        array_of_elements_with_0_derivative_, frequencies_, depth_):
    contours_without_excess_peaks = []
    for i in range(len(array_of_maximums)):
        peaks_to_merge = []
        xmin = min(array_of_x_coordinates[i])
        peaks_in_one_cnt = array_of_maximums[i]
        if (len(peaks_in_one_cnt != 0)):
            indices = find_indices(peaks_in_one_cnt, array_of_minimums[i])
            # в контуре 1 пик
            if (len(indices) == 1):
                num_peak = peaks_in_one_cnt[0]
                if (num_peak in array_of_elements_with_0_derivative_[i][0]):
                    ind1, ind2 = array_of_elements_with_0_derivative_[i][0][num_peak]
                    xmax, ymax, ymin = select_xmax_ymax_from_2_cnt_index(array_of_cnts, i, ind1, ind2)
                    temp_cnt = [xmin, ymin, xmax, ymax]
                else:
                    xmax = array_of_cnts[i][num_peak][0][0]
                    ymax = array_of_cnts[i][num_peak][0][1]
                    ymin = ymax
                    xmax, ymax = min(xmax, len(frequencies_) - 1), min(ymax, len(depth_) - 1)
                    temp_cnt = [xmin, ymin, xmax, ymax]
                peaks_to_merge.append(temp_cnt)
                contours_without_excess_peaks.append(peaks_to_merge)
                peaks_to_merge = []
            else:
                for j in range(len(peaks_in_one_cnt)):
                    num_peak = peaks_in_one_cnt[j]
                    if (num_peak in array_of_elements_with_0_derivative_[i][0]):
                        ind1, ind2 = array_of_elements_with_0_derivative_[i][0][num_peak]
                        x_current_peak, y_current_peak, ymin = select_xmax_ymax_from_2_cnt_index(array_of_cnts, i, ind1,
                                                                                                 ind2)
                    else:
                        x_current_peak = array_of_cnts[i][num_peak][0][0]
                        y_current_peak = array_of_cnts[i][num_peak][0][1]
                        ymin = y_current_peak
                    trh_difference = x_current_peak / 4
                    # нет минимума слева
                    if (indices[j][0] == -1):
                        num_min_right = array_of_minimums[i][indices[j][1]]
                        x_right = array_of_x_coordinates[i][num_min_right]
                        if (x_current_peak - x_right < trh_difference):
                            temp_cnt = [xmin, ymin, x_current_peak, y_current_peak]
                            peaks_to_merge.append(temp_cnt)
                        else:
                            temp_cnt = [xmin, ymin, x_current_peak, y_current_peak]
                            peaks_to_merge.append(temp_cnt)
                            contours_without_excess_peaks.append(peaks_to_merge)
                            peaks_to_merge = []
                    # нет минимума справа
                    elif (indices[j][1] == -1):
                        num_min_left = array_of_minimums[i][indices[j][0]]
                        x_left = array_of_x_coordinates[i][num_min_left]
                        if (x_current_peak - x_left < trh_difference):
                            temp_cnt = [xmin, ymin, x_current_peak, y_current_peak]
                            peaks_to_merge.append(temp_cnt)
                        else:
                            if (len(peaks_to_merge) != 0):
                                contours_without_excess_peaks.append(peaks_to_merge)
                            peaks_to_merge = []
                            temp_cnt = [xmin, ymin, x_current_peak, y_current_peak]
                            peaks_to_merge.append(temp_cnt)
                    # есть минимум и слева и справа
                    else:
                        num_min_left = array_of_minimums[i][indices[j][0]]
                        num_min_right = array_of_minimums[i][indices[j][1]]
                        x_left = array_of_x_coordinates[i][num_min_left]
                        x_right = array_of_x_coordinates[i][num_min_right]
                        if (x_current_peak - x_left < trh_difference) and (x_current_peak - x_right < trh_difference):
                            temp_cnt = [xmin, ymin, x_current_peak, y_current_peak]
                            peaks_to_merge.append(temp_cnt)
                        elif (x_current_peak - x_left >= trh_difference) and (
                                x_current_peak - x_right < trh_difference):
                            if (len(peaks_to_merge) != 0):
                                contours_without_excess_peaks.append(peaks_to_merge)
                            peaks_to_merge = []
                            temp_cnt = [xmin, ymin, x_current_peak, y_current_peak]
                            peaks_to_merge.append(temp_cnt)
                        elif (x_current_peak - x_left < trh_difference) and (
                                x_current_peak - x_right >= trh_difference):
                            temp_cnt = [xmin, ymin, x_current_peak, y_current_peak]
                            peaks_to_merge.append(temp_cnt)
                            contours_without_excess_peaks.append(peaks_to_merge)
                            peaks_to_merge = []
                        else:
                            if (len(peaks_to_merge) != 0):
                                contours_without_excess_peaks.append(peaks_to_merge)
                            peaks_to_merge = []
                            temp_cnt = [xmin, ymin, x_current_peak, y_current_peak]
                            peaks_to_merge.append(temp_cnt)
                            contours_without_excess_peaks.append(peaks_to_merge)
                            peaks_to_merge = []
                if (len(peaks_to_merge) != 0):
                    contours_without_excess_peaks.append(peaks_to_merge)
        # пики в контуре не найдены, поэтому весь контур выделяется в один объект
        else:
            peaks_in_one_cnt = array_of_cnts[i]
            arr = []
            for peak in peaks_in_one_cnt:
                peak = peak[0]
                arr.append(peak)
            xmax = arr[0][0]
            ymax = ymin = arr[0][1]
            xmin = xmax
            for i in range(len(arr)):
                xmax = max(xmax, arr[i][0])
                ymax = max(ymax, arr[i][1])
                xmin = min(xmin, arr[i][0])
                ymin = min(ymin, arr[i][1])
            temp_cnt = [xmin, ymin, xmax, ymax]
            peaks_to_merge.append(temp_cnt)
            contours_without_excess_peaks.append(peaks_to_merge)
    return contours_without_excess_peaks


# выделение общего прямоугольника для контуров
def select_common_rectangle(array_of_cnts):
    xmin = array_of_cnts[0][0]
    ymin = array_of_cnts[0][1]
    xmax = array_of_cnts[0][2]
    ymax = array_of_cnts[0][3]
    for i in range(1, len(array_of_cnts)):
        xmin = min(xmin, array_of_cnts[i][0])
        ymin = min(ymin, array_of_cnts[i][1])
        xmax = max(xmax, array_of_cnts[i][2])
        ymax = max(ymax, array_of_cnts[i][3])
    temp_cnt = [xmin, ymin, xmax, ymax]
    return temp_cnt


# Группировка контуров в объединённые прямоугольники
def group_contours_to_rectangles(contours_list):
    return [select_common_rectangle(cnt) for cnt in contours_list]


# выделение значимых пиков
def select_significant_peaks(array_of_cnts, frequencies_, depth_):
    if array_of_cnts is None or len(array_of_cnts) == 0:
        raise ValueError('Массив пуст, объекты не определены')
    array_of_x_coordinates_of_cnts = get_x_coordinates_from_cnt(array_of_cnts)
    negated_array_of_x_coordinates_of_cnts = get_negated_array(array_of_x_coordinates_of_cnts)
    array_of_maximums_of_cnt = find_peaks_function(array_of_x_coordinates_of_cnts)
    array_of_minimums_of_cnt = find_peaks_function(negated_array_of_x_coordinates_of_cnts)
    array_of_elements_with_0_derivative = get_array_of_elements_with_0_derivative(array_of_maximums_of_cnt,
                                                                                  array_of_x_coordinates_of_cnts)
    array_of_contours_without_excess_peaks = remove_excess_peaks(array_of_cnts, array_of_maximums_of_cnt,
                                                                 array_of_minimums_of_cnt,
                                                                 array_of_x_coordinates_of_cnts,
                                                                 array_of_elements_with_0_derivative, frequencies_,
                                                                 depth_)
    new_contours = group_contours_to_rectangles(array_of_contours_without_excess_peaks)
    return new_contours


# удаление очень маленьких и слабых шумов
def delete_small_contours(array_of_cnts, aps_data_, min_square=10):
    if array_of_cnts == None or len(array_of_cnts) == 0:
        raise ValueError('Массив пуст, объекты не определены')
    points = []
    common_mean = np.percentile(aps_data_, 40)
    for t in array_of_cnts:
        xmin_matr, ymin_matr, xmax_matr, ymax_matr = t
        w = xmax_matr - xmin_matr + 1
        h = ymax_matr - ymin_matr + 1
        sqr = w * h
        temp_data = aps_data_[int(ymin_matr):int(ymax_matr) + 1, int(xmin_matr):int(xmax_matr) + 1]
        mean = np.mean(temp_data) if temp_data.size > 0 else 0
        if sqr > min_square and mean > common_mean:
            points.append([xmin_matr, ymin_matr, xmax_matr, ymax_matr])
    return points


# нахождение пересечений одновременно и по х, и по у, меньший по полощади удаляется
def find_contour_indexes_to_keep(array_of_cnts):
    if array_of_cnts == None or len(array_of_cnts) == 0:
        raise ValueError('Массив пуст, объекты не определены')
    points = np.array(array_of_cnts)
    to_keep = set()
    for i, (xmin1, ymin1, xmax1, ymax1) in enumerate(points):
        sqr1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        keep_i = True
        for j, (xmin2, ymin2, xmax2, ymax2) in enumerate(points):
            if i == j:
                continue
            # условие пересечения по X и по Y
            x_overlap = not (xmax1 < xmin2 or xmax2 < xmin1)
            y_overlap = not (ymax1 < ymin2 or ymax2 < ymin1)
            if x_overlap and y_overlap:
                sqr2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
                # оставляем только больший по площади контур
                if sqr2 > sqr1:
                    keep_i = False
                    break  # нет смысла дальше проверять, т.к. этот уже меньше
        if keep_i:
            to_keep.add(i)
    return sorted(to_keep)


# отрисовка новых контуров
def extract_contours_coordinates(array_of_cnt, cnts_with_x_y_intersections):
    if array_of_cnt == None or len(array_of_cnt) == 0:
        raise ValueError('Массив пуст, объекты не определены')
    array_of_cnt = np.array(array_of_cnt)
    new_points = []
    for index in cnts_with_x_y_intersections:
        if index < 0 or index >= len(array_of_cnt):
            raise IndexError(f'Индекс {index} вне диапазона массива')
        xmin_matr = array_of_cnt[index, 0]
        xmax_matr = array_of_cnt[index, 2]
        ymin_matr = array_of_cnt[index, 1]
        ymax_matr = array_of_cnt[index, 3]
        new_points.append([xmin_matr, ymin_matr, xmax_matr, ymax_matr])

    return np.array(new_points)


# нахождение пересечений по y
def find_y_intersections(array_of_cnts):
    if array_of_cnts is None or array_of_cnts.size == 0:
        raise ValueError('Массив пуст, объекты не определены')

    intersections = []
    for index, (x_min, y_min, x_max, y_max) in enumerate(array_of_cnts):
        intersecting_indices = [
            i for i, (_, y_min2, _, y_max2) in enumerate(array_of_cnts)
            if i != index and (
                    (y_min2 <= y_min <= y_max2) or
                    (y_min2 >= y_min and y_max >= y_min2)
            )
        ]
        intersection = [index] + intersecting_indices
        intersections.append(intersection)
    return intersections


# нахождение всех персечений (условно если 4 пересекается с 9, а 9 с 11, то 4 пересекается с 9 и 11)
def find_all_intersections(n, intersections_):
    added_indexes = []
    prevArr = intersections_[n]
    array = intersections_[n].copy()
    added_indexes.append(n)
    for i in range(1, len(prevArr)):
        k = prevArr[i]
        if k not in added_indexes:
            added_indexes.append(k)
            for element in intersections_[k]:
                array.append(element)
    array = list(dict.fromkeys(array))
    while (array != prevArr):
        prevArr = array
        for i in range(1, len(prevArr)):
            k = prevArr[i]
            if k not in added_indexes:
                added_indexes.append(k)
                for element in intersections_[k]:
                    array.append(element)
        array = list(dict.fromkeys(array))
    return (array)


# обьединение пересечений в один объект
def merge_contours(array_of_cnt_, all_intersections_list_):
    if array_of_cnt_ is not None and array_of_cnt_.size != 0:
        mer_points = []
        added_idx = []
        for i in range(len(all_intersections_list_)):
            if i not in added_idx:
                added_idx.append(i)
                intersection = all_intersections_list_[i]
                xmin_matr = array_of_cnt_[i, 0]
                xmax_matr = array_of_cnt_[i, 2]
                ymin_matr = array_of_cnt_[i, 1]
                ymax_matr = array_of_cnt_[i, 3]
                for j in range(1, len(intersection)):
                    element = intersection[j]
                    added_idx.append(element)
                    xmin_matr = min(xmin_matr, array_of_cnt_[element, 0])
                    xmax_matr = max(xmax_matr, array_of_cnt_[element, 2])
                    ymin_matr = min(ymin_matr, array_of_cnt_[element, 1])
                    ymax_matr = max(ymax_matr, array_of_cnt_[element, 3])
                temp_cnt = [xmin_matr, ymin_matr, xmax_matr, ymax_matr]
                mer_points.append(temp_cnt)
        return mer_points
    else:
        raise ValueError('Массив пуст, объекты не определены')


# ограничение на начало обьекта
def add_noise_start_limit(array_of_cnt_):
    if array_of_cnt_ is not None and len(array_of_cnt_) != 0:
        new_xmin_points = []
        t_points = array_of_cnt_
        for i in range(0, len(t_points)):
            xmin_matr = t_points[i][0]
            xmax_matr = t_points[i][2]
            ymin_matr = t_points[i][1]
            ymax_matr = t_points[i][3]
            if (xmin_matr < constants_store.constants.TRH_XMIN_MATR):
                temp = [xmin_matr, ymin_matr, xmax_matr, ymax_matr]
                new_xmin_points.append(temp)
        return new_xmin_points
    else:
        raise ValueError('Массив пуст, объекты не определены')


# ограничение на конец обьекта
def add_noise_end_limit(array_of_cnt_):
    if array_of_cnt_ is not None and len(array_of_cnt_) != 0:
        xmax_restrict_points = []
        t_points = array_of_cnt_
        for i in range(0, len(t_points)):
            xmin_matr = t_points[i][0]
            xmax_matr = t_points[i][2]
            ymin_matr = t_points[i][1]
            ymax_matr = t_points[i][3]
            if (xmax_matr >= constants_store.constants.TRH_XMAX_MATR):
                temp = [xmin_matr, ymin_matr, xmax_matr, ymax_matr]
                xmax_restrict_points.append(temp)
        return xmax_restrict_points
    else:
        raise ValueError('Массив пуст, объекты не определены')


# ограничение по площади
def add_sqr_limit(array_of_cnt_, frequencies_, depth_):
    if array_of_cnt_ is not None and len(array_of_cnt_) != 0:
        sqr_restrict_points = []
        max_width = 0
        t_points = array_of_cnt_
        for i in range(0, len(t_points)):
            xmin_matr = t_points[i][0]
            xmax_matr = t_points[i][2]
            ymin_matr = t_points[i][1]
            ymax_matr = t_points[i][3]
            w = xmax_matr - xmin_matr + 1
            h = ymax_matr - ymin_matr + 1
            max_width = max(max_width, w)
            sqr = w * h
            if (sqr > constants_store.constants.TRH_SQR):
                temp = [xmin_matr, ymin_matr, xmax_matr, ymax_matr]
                sqr_restrict_points.append(temp)
        return sqr_restrict_points, max_width
    else:
        raise ValueError('Массив пуст, объекты не определены')


# ограничение по ширине и интенсивности
def add_width_intensity_limit(array_of_cnt_, max_width_, aps_data_):
    if array_of_cnt_ is not None and len(array_of_cnt_) != 0:
        width_restrict_points = []
        trh_width_10 = max_width_ * 0.05
        for i in range(0, len(array_of_cnt_)):
            xmin_matr = array_of_cnt_[i][0]
            xmax_matr = array_of_cnt_[i][2]
            ymin_matr = array_of_cnt_[i][1]
            ymax_matr = array_of_cnt_[i][3]
            w = xmax_matr - xmin_matr + 1
            temp_data = aps_data_[ymin_matr:ymax_matr + 1, xmin_matr:xmax_matr + 1]
            common_mean = np.percentile(aps_data_, 45)
            mean = 0
            if (len(temp_data) > 0):
                mean = np.mean(temp_data)

            if (w > trh_width_10) and (mean > common_mean):
                temp = [xmin_matr, ymin_matr, xmax_matr, ymax_matr]
                width_restrict_points.append(temp)
        return width_restrict_points
    else:
        raise ValueError('Массив пуст, объекты не определены')


# расширение обьектов по глубине на 1 (если на соседней глубине есть объект, то расширения не происходит)
def wide_contours(array_of_cnts, depth_):
    if array_of_cnts is None or len(array_of_cnts) == 0:
        raise ValueError('Массив пуст, объекты не определены')
    widen_contours = []
    array_of_cnts.sort(key=lambda x: x[1])
    n = len(array_of_cnts)
    for i, cnt in enumerate(array_of_cnts):
        xmin, ymin, xmax, ymax = map(int, cnt)
        # Проверка на соседние объекты по глубине
        prev_touch = i > 0 and (ymin - 1 == array_of_cnts[i - 1][3])
        next_touch = i < n - 1 and (ymax + 1 == array_of_cnts[i + 1][1])
        if prev_touch or next_touch:
            temp_cnt = [xmin, ymin, xmax, ymax]
        else:
            new_ymin = max(ymin - 1, 0)
            new_ymax = min(ymax + 1, len(depth_) - 1)
            temp_cnt = [xmin, new_ymin, xmax, new_ymax]
        widen_contours.append(temp_cnt)
    return widen_contours


# подсчет амплитуды сигналов
def calculate_amplitudes_of_signal(points, frequencies, depth, aps_data_):
    amplitudes = [0 for i in range(len(points))]
    for i in range(len(points)):
        xmin_matr = np.abs(frequencies - points[i]['x_min']).argmin()
        xmax_matr = np.abs(frequencies - points[i]['x_max']).argmin()
        ymin_matr = np.abs(depth - points[i]['y_min']).argmin()
        ymax_matr = np.abs(depth - points[i]['y_max']).argmin()
        temp = aps_data_[ymin_matr:ymax_matr + 1, xmin_matr:xmax_matr + 1]
        perc99 = np.percentile(temp, 99)
        amplitudes[i] = perc99
    return amplitudes


# определение типа шума
def determiningNoise(points, frequencies, depth, language_):
    object_type = ["" for i in range(len(points))]
    for i in range(len(points)):
        dd = np.abs(depth - points[i]['y_max']).argmin() - np.abs(depth - points[i]['y_min']).argmin() + 1
        df = np.abs(frequencies - points[i]['x_max']).argmin() - np.abs(frequencies - points[i]['x_min']).argmin() + 1
        f1 = frequencies[np.abs(frequencies - points[i]['x_min']).argmin()]
        f2 = frequencies[np.abs(frequencies - points[i]['x_max']).argmin()]
        if df > dd and f1 > 0:
            object_type[i] = "Reservoir"
            if (language_ == "r"):
                object_type[i] = "Поток по пласту"
        elif df < dd and f1 > 1:
            object_type[i] = "Chanelling"
            if (language_ == "r"):
                object_type[i] = "Заколонная циркуляция"
        else:
            if f2 > 10:
                object_type[i] = "Reservoir"
                if (language_ == "r"):
                    object_type[i] = "Поток по пласту"
            else:
                object_type[i] = "Borehole"
                if (language_ == "r"):
                    object_type[i] = "Буровая колонна"
    return object_type


def createTable(rects, object_type, arr_perc99):
    table = []
    for i in range(len(rects)):
        top_depth = int(round(rects[i]['y_min'], 0))
        bottom_depth = int(round(rects[i]['y_max'], 0))
        low_freq = float(round(rects[i]['x_min'], 1))
        high_freq = float(round(rects[i]['x_max'], 1))
        freq = str(low_freq) + '-' + str(high_freq)
        ampl_db_99 = int(round(arr_perc99[i], 0))
        flow_type = object_type[i]
        temp_table_str = [top_depth, bottom_depth, freq, ampl_db_99, flow_type]
        table.append(temp_table_str)
    return table


def upload_report_to_excel(points, frequencies_, depth_, aps_data_, language_, file_path):
    arr_perc99 = calculate_amplitudes_of_signal(points, frequencies_, depth_, aps_data_)
    noise_type = determiningNoise(points, frequencies_, depth_, language_)
    table = createTable(points, noise_type, arr_perc99)
    if language_ == 'r':
        df = pd.DataFrame(table, columns=["Кровля, м", "Подошва, м", "Частотный диапазон, кГц",
                                          "Амплитуда, дб",
                                          "Характеристика типа шума"])
    else:
        df = pd.DataFrame(table, columns=[" Top, m", "Bottom, m   ", "Frequency range, kHz ",
                                          "Amplitude, dB",
                                          "Flow type"])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = df.set_index(np.arange(1, len(df) + 1))
    df.to_excel(file_path, index=True)


def createRects(points, frequencies_, depth_):
    rects = []
    for i in range(len(points)):
        top_depth = float(depth_[points[i][1]])
        bottom_depth = float(depth_[points[i][3]])
        low_freq = float(frequencies_[points[i][0]])
        high_freq = float(frequencies_[points[i][2]])
        temp_table_str = [low_freq, top_depth, high_freq, bottom_depth]
        rects.append(temp_table_str)
    return rects


# создание таблицы с интервалом глубин
def create_depth_table(points, depth_):
    table = []
    for i in range(len(points)):
        top_depth = int(round(depth_[points[i][1]]))
        bottom_depth = int(round(depth_[points[i][3]]))
        temp_table_str = [top_depth, bottom_depth]
        table.append(temp_table_str)
    return table


# выгрузка интервалов глубин в las-файл
def upload_intervals_to_las_file(array_of_cnts, depth_, file_path):
    # массив со значениями 0/1 в зависмости от того есть ли обьект на этой глубине
    identificator_array = np.zeros(len(depth_))
    for cnt in array_of_cnts:
        ymin = np.abs(depth_ - cnt['y_min']).argmin()
        ymax = np.abs(depth_ - cnt['y_max']).argmin()
        for i in range(ymin, ymax + 1):
            identificator_array[i] = 1
    las = lasio.LASFile()
    las.params['DEPTH'] = lasio.HeaderItem('DEPTH')
    las.params['LOGLAYERS'] = lasio.HeaderItem('LOGLAYERS')
    las.append_curve('DEPTH', depth_, unit='m')
    las.append_curve('LOGLAYERS', identificator_array)
    las.write(file_path)
