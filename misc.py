"""
This module includes some basic functionaly that can be used for every other file in the package.
"""
import numpy as np


def weighted_gravity_centers(diffArray, firstFrame=1, lastFrame=None, threshold=0):
    """
    Computes the x,y coordinates of the laser spot from a diffArray. The position is calculated with a
    weighted gravity center. First and last frame of the video can be specified.
    The pixels below the threshold are not considered in the computation of the gravity center.
    :param diffArray: Diff array of the video. Shape (n_frames-1, rows, cols).
    :param firstFrame: First frame to consider in the video. The frames [0, firstFrame) in diffArray won't be processed.
    :param lastFrame:  Last frame to consider in the video. The frames [lastFrame, n_frames) in diffArray won't be processed.
    If lastFrame is equal to None, then lastFrame will be n_frames.
    :param threshold: Threshold to disable some pixels in the computation of the gravity center.
    :return: Coordinates x,y of the laser spot.
    """
    horizontalIndexes = np.tile(np.arange(1,diffArray.shape[2]+1),(diffArray.shape[1],1))
    verticalIndexes = np.tile(np.arange(1,diffArray.shape[1]+1).reshape(-1,1),(1,diffArray.shape[2]))

    if lastFrame is None:
        lastFrame = diffArray.shape[0]

    diffThreshold = diffArray.copy()
    diffThreshold[diffThreshold < threshold] = 0

    sumVector = diffThreshold[firstFrame-1:lastFrame].sum(axis=(1,2))
    sumVector = sumVector.astype(np.float_)
    normDiff = diffThreshold[firstFrame-1:lastFrame] / sumVector[:,np.newaxis, np.newaxis]
    x = (normDiff*horizontalIndexes).sum(axis=(1,2)) - 1
    y = (normDiff*verticalIndexes).sum(axis=(1,2)) - 1

    return x,y

def fix_nan(array):
    """
    Removes the NaN values in a position array by using the previous valid value until
    there are not more NaN values. Useful because in some cases the difference between
    two frames is equal to 0 and the algorithm is not able to compute the position.
    :param array: Array where want to remove NaN values.
    :return: Array without NaN values.
    """
    while np.isnan(array).sum():
        indexNaN = np.where(np.isnan(array))[0]
        indexNaNMinus = indexNaN -1
        array[indexNaN] = array[indexNaNMinus]

    return array