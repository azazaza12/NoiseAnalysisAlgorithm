class Constants:
    def __init__(self, trh_sqr, trh_xmin_matr, trh_xmax_matr,
                 medfilt_window_one_1, medfilt_window_one_2,
                 medfilt_window_two_1, medfilt_window_two_2):
        self.TRH_SQR = trh_sqr
        self.TRH_XMIN_MATR = trh_xmin_matr
        self.TRH_XMAX_MATR = trh_xmax_matr
        self.MEDFILT_WINDOW_ONE_1 = medfilt_window_one_1
        self.MEDFILT_WINDOW_ONE_2 = medfilt_window_one_2
        self.MEDFILT_WINDOW_TWO_1 = medfilt_window_two_1
        self.MEDFILT_WINDOW_TWO_2 = medfilt_window_two_2


def initialize_constants(frequencies_, depth_, channel_type_):
    if channel_type_ == 'LF':
        trh_sqr = 30
        trh_xmin_matr = 400
        trh_xmax_matr = (frequencies_ > 1).tolist().index(True)
    else:
        trh_sqr = 30
        trh_xmin_matr = 150
        trh_xmax_matr = (frequencies_ > 7).tolist().index(True)

    medfilt_window_one_1 = round(len(depth_) / 4)
    if medfilt_window_one_1 % 2 == 0:
        medfilt_window_one_1 += 1

    medfilt_window_one_2 = 1
    medfilt_window_two_1 = 1
    medfilt_window_two_2 = 9

    return Constants(
        trh_sqr, trh_xmin_matr, trh_xmax_matr,
        medfilt_window_one_1, medfilt_window_one_2,
        medfilt_window_two_1, medfilt_window_two_2
    )
