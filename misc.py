"""
This module includes some basic functionaly that can be used for every other file in the package.
"""
import numpy as np


def weighted_gravity_centers(diff_array, first_frame=1, last_frame=None, threshold=0):
    """
    Computes the column,row coordinates of the laser spot from a diffArray. The position is calculated with a
    weighted gravity center. First and last frame of the video can be specified.
    The pixels below the threshold are not considered in the computation of the gravity center.
    :param diff_array: Diff array of the video. Shape (n_frames-1, rows, cols).
    :param first_frame: First frame to consider in the video. The frames [0, firstFrame) in diffArray won't be processed.
    :param last_frame:  Last frame to consider in the video. The frames [lastFrame, n_frames) in diffArray won't be processed.
    If lastFrame is equal to None, then lastFrame will be n_frames.
    :param threshold: Threshold to disable some pixels in the computation of the gravity center.
    :return: Coordinates column,row of the laser spot.
    """
    horizontal_indexes = np.tile(np.arange(1, diff_array.shape[2] + 1), (diff_array.shape[1], 1))
    vertical_indexes = np.tile(np.arange(1, diff_array.shape[1] + 1).reshape(-1, 1), (1, diff_array.shape[2]))

    if last_frame is None:
        last_frame = diff_array.shape[0]

    diff_threshold = diff_array.copy()
    diff_threshold[diff_threshold < threshold] = 0

    sum_vector = diff_threshold[first_frame - 1:last_frame].sum(axis=(1, 2))
    sum_vector = sum_vector.astype(np.float_)
    norm_diff = diff_threshold[first_frame - 1:last_frame] / sum_vector[:, np.newaxis, np.newaxis]
    column = (norm_diff*horizontal_indexes).sum(axis=(1,2)) - 1
    row = (norm_diff*vertical_indexes).sum(axis=(1,2)) - 1

    return column,row

def fix_nan(array):
    """
    Removes the NaN values in a position array by using the previous valid value until
    there are not more NaN values. Useful because in some cases the difference between
    two frames is equal to 0 and the algorithm is not able to compute the position.
    :param array: Array where want to remove NaN values.
    :return: Array without NaN values.
    """
    while np.isnan(array).sum():
        index_nan = np.where(np.isnan(array))[0]
        index_nan_previous = index_nan - 1
        array[index_nan] = array[index_nan_previous]

    return array