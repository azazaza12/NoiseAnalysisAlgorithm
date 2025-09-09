from tkinter import messagebox
import lasio
import numpy as np
from scipy.signal import medfilt
import functions
from constants import initialize_constants
import constants_store


def get_data(data_path_):
    try:
        las = lasio.read(data_path_)
        num_of_channels = functions.get_count_num_channels(las)
        channel_type = functions.get_channel_type(las)
        depth = las.depth_m
        if channel_type == 'LF':
            frequencies = np.linspace(0.01, 4.883, num_of_channels)
        else:
            frequencies = np.linspace(0.114, 58.594, num_of_channels)
        aps_data = las.data[:, 1:num_of_channels + 1]
        aps_data = np.nan_to_num(aps_data)
        constants_store.constants = initialize_constants(frequencies, depth, channel_type)
        return frequencies, depth, aps_data
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при загрузке файла или обработке данных:\n{str(e)}")
        return [], [], []


def detect_objects(frequencies, depth, aps_data):
    try:
        #добавление строк во избежание краевых эффектов
        count_new_up_lines = round(constants_store.constants.MEDFILT_WINDOW_ONE_1 / 4)
        new_lines_aps_data = functions.add_new_lines(aps_data, count_new_up_lines)
        #медианная фильтрация вдоль оси глубины
        medfilt_aps_data = medfilt(new_lines_aps_data, [constants_store.constants.MEDFILT_WINDOW_ONE_1,
                                                        constants_store.constants.MEDFILT_WINDOW_ONE_2])
        sub_lines_aps_data = new_lines_aps_data - medfilt_aps_data
        #медианная фильтрация вдоль оси частоты
        med_sub_lines_aps_data = medfilt(sub_lines_aps_data, [constants_store.constants.MEDFILT_WINDOW_TWO_1,
                                                              constants_store.constants.MEDFILT_WINDOW_TWO_2])
        median_drift = med_sub_lines_aps_data[count_new_up_lines:-count_new_up_lines, :]

        blured_data = functions.blur_data(median_drift)
        binarised_aps_data = functions.binarisation(aps_data, blured_data)
        dilated_data = functions.dilate_data(binarised_aps_data)

        array_of_contours = functions.find_contours(dilated_data)
        significant_contours = functions.select_significant_peaks(array_of_contours, frequencies, depth)
        points_deleted_small_cnts = functions.delete_small_contours(significant_contours, aps_data)
        #поиск пересечений по глубине и частоте, и удаление
        points_with_intersec_x_y = functions.find_contour_indexes_to_keep(points_deleted_small_cnts)
        points_deletion_intersec_x_y = functions.extract_contours_coordinates(points_deleted_small_cnts,
                                                                                     points_with_intersec_x_y)
        #поиск взаимных пересечений по глубине, и объединение
        intersections = functions.find_y_intersections(points_deletion_intersec_x_y)
        all_intersections_list = []
        for i in range(len(intersections)):
            temp_arr = functions.find_all_intersections(i, intersections)
            all_intersections_list.append(temp_arr)
        merged_points = functions.merge_contours(points_deletion_intersec_x_y, all_intersections_list)
        poins_after_xmin_restrict = functions.add_noise_start_limit(merged_points)
        poins_after_xmax_restrict = functions.add_noise_end_limit(poins_after_xmin_restrict)
        poins_after_sqr_restrict, max_width = functions.add_sqr_limit(poins_after_xmax_restrict, frequencies, depth)
        poins_after_width_intensity_restrict = functions.add_width_intensity_limit(poins_after_xmax_restrict, max_width,
                                                                                   aps_data)
        poins_after_wide = functions.wide_contours(poins_after_width_intensity_restrict, aps_data)
        rects = functions.createRects(poins_after_wide, frequencies, depth)
        return rects
    except Exception as e:
        return None
