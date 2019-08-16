"""
    This file contains multiple auxiliary functionality to work with sets of observations (laser spot positions).

    There are two main classes.
        - ObservationSet which contains the laser sport positions for multiple videos including the temporal window
        information.
        - ObservationROISet which implements the same functionality of ObservationSet, but is needed to work with
        the region of interest (ROI) of the videos.

    There are other classes useful for the D-Markov machine classifier:
        - SymbolizationType: to select the type of symbolization method.
        - DivisionOrder: to select the division order or equal frequency and equal frequency no bounds symbolization.
        - EqualWidthLimits: Implements equal width symbolization.
        - EqualFrequencyLimits: Implements equal frequency symbolization.
        - EqualFrequencyLimitsNoBounds: Implements equal frequency no bounds symbolization.
        - SymbolSet: This class represents a set of symbol, in a similar way to ObservationSet representing a
        set of laser spot positions. A SymbolSet can be generated from an ObservationSet using a symbolization method.

"""

import numpy as np

import warnings
import os
import misc
import pickle
import fnmatch
from enum import Enum
import random

import sys

class ObservationSet(object):
    """
    An observation set is a collection of laser spot positions for each video. It contains some useful attributes:

    - obs: A 3-D array (n_videos x 2 x max_length_video). This attribute contains the laser spot positions in a 3-D array.
        * The first axis contains the index of the video.
        * The second axis selects betwen the columns axis [0] and rows axis [1] components of the position.
        * The third axis selects the temporal moment of the position. The third axis has the same length as the length
        of the longest video (max_length_video). The videos with less length have a padding of -1.
    - length_vector: A 1-D array containing the length of each video.
    - names_vector: a 1-D array containing the name of each video.
    - n_window: Number of temporal windows.
    - window_start: A 1-D array containing the index of the starting frame of each temporal window.
    - window_limits: A 1-D array containing the index of the last frame of each temporal window.
    """
    def __init__(self, observations, length_vector, names_vector, n_window=1):
        self.obs = observations
        self.length_vector = length_vector
        self.names_vector = names_vector

        self.n_window = n_window
        if n_window > 0:
            self._update_window_limits()

    def __getitem__(self, item):
        return self.obs[item]

    def __repr__(self):
        return "ObservationSet with " + str(self.num_observations()) + "x" + str(self.obs.shape[2]) +  " obs. [" + str(self.num_windows()) + " window]"

    def synchronize_average(self, inplace=True, center_row=16, center_col=16):
        """
        Synchronize the observations position by moving the points so
        the average on the vertical and horizontal axis are the same
        for all the observations. If some observations lies outside the range
        [0,32] after the transformation, a ValueError is raised.

        :return: Synchronized observations.
        """

        obs_set = self if inplace else self.copy()

        obs_set._move_center(center_col, center_row)

        col_bounds, row_bounds = obs_set.space_bounds()

        if col_bounds[0] < 0 or col_bounds[1] > 32 or row_bounds[0] < 0 or row_bounds[1] > 32:
            raise ValueError("Synchronization didn't suceed.")

        return obs_set

    def _move_center(self, center_col, center_row):
        """
        Move the average center of the observations to a given center.
        :param center_col:
        :param center_row:
        :return:
        """
        for n in range(0, self.num_observations()):
            len_obs = self.length_vector[n]
            mean_row = np.average(self.obs[n,1,:len_obs])
            diff_row = center_row - mean_row
            self.obs[n,1,:len_obs] += diff_row

            mean_col = np.average(self.obs[n,0,:len_obs])
            diff_col = center_col - mean_col
            self.obs[n,0,:len_obs] += diff_col

    def gaussian_noise(self, std_col=0.02, std_row=0.02, inplace=True, seed=0):
        """
        Adds independent gaussian noise in both axis. The standard deviation of the noise of columns axis and rows axis can be
        setted using std_col and std_row. If adding gaussian noise makes the observations go out of bounds, numpy.clip is called
        to make the observations lie in [0,32).
        :param std_col: Columns axis Gaussian standard deviation.
        :param std_row: Rows axis Gaussian standard deviation.
        :param inplace: Make changes inplace.
        :return: Noised observations.
        """
        np.random.seed(seed)
        obs_set = self if inplace else self.copy()

        for n in range(0,obs_set.num_observations()):
            len_obs = obs_set.length_vector[n]
            noise_col = np.random.normal(0, std_col, len_obs)
            noise_row = np.random.normal(0, std_row, len_obs)

            obs_set.obs[n,0,:len_obs] += noise_col
            obs_set.obs[n,1,:len_obs] += noise_row

        col_bounds, row_bounds = obs_set.space_bounds()

        if col_bounds[0] < 0 or col_bounds[1] > 32 or row_bounds[0] < 0 or row_bounds[1] > 32:
            obs_set._clip()
            warnings.warn("Values clipped adding gaussian noise.")

        return obs_set

    def _clip(self, col_min=0, col_max=31.99, row_min=0, row_max=31.99, inplace=True):
        """
        Clips the observation so they lie within [x_min, x_max] and [y_min, y_max].
        :param col_min: Columns axis min value.
        :param col_max: Columns axis max value.
        :param row_min: Rows axis min value.
        :param row_max: Rows axis max value.
        :param inplace:
        :return:
        """
        obs_set = self if inplace else self.copy()

        for n in range(0,self.num_observations()):
            len_obs = self.length_vector[n]
            obs_set.obs[n,0,:len_obs] = np.clip(obs_set.obs[n,0,:len_obs], col_min, col_max)
            obs_set.obs[n,1,:len_obs] = np.clip(obs_set.obs[n,1,:len_obs], row_min, row_max)

        return obs_set

    def mean_pos_col(self):
        """
        Returns the mean of the observations in the columns axis.
        :return:
        """
        sum = 0
        for n in range(0, self.num_observations()):
            sum += self.obs[:,0,:].mean()
        return sum / self.num_observations()

    def mean_pos_row(self):
        """
        Returns the mean of the observations in the rows axis.
        :return:
        """
        sum = 0
        for n in range(0, self.num_observations()):
            sum += self.obs[:,1,:].mean()
        return sum / self.num_observations()

    def crop_to_min_index(self, inplace=True):
        """
        Crops the observations to the length of the shortest observation. Destructive operation.
        :param inplace:
        :return:
        """
        obs_set = self if inplace else self.copy()
        min_index = obs_set.min_length()
        obs_set.obs = obs_set.obs[:, :, :min_index]

        obs_set.length_vector[:] = obs_set.obs.shape[2]
        self._update_window_limits()
        return obs_set

    def crop_to_max_index(self, inplace=True):
        """
        Crops the observations to the length of the current largest observation. This is usually useful when making changes
        to the ObservationSet, such as removing observations, because the current observations could be shorter than
        the largest original observation.
        :param inplace:
        :return:
        """
        obs_set = self if inplace else self.copy()
        max_index = obs_set.length_vector.max()

        if obs_set.obs.shape[2] > max_index:
            obs_set.obs = obs_set.obs[:, :, :max_index]

        self._update_window_limits()
        return obs_set

    def crop_between_indices(self, low, high, inplace=False):
        """
        Crops the length (time) of the observations between the index low (inclusive) and between (exclusive). This
        operation invalidates the existence of windows because arbitrary indexes can be selected for low and high.
        :param low: Low index.
        :param high: Higher index.
        :param inplace: Make changes inplace.
        :return:
        """
        obs_set = self._crop_between_indices(low, high, inplace)
        obs_set._invalidate_window()
        return obs_set

    def _crop_between_indices(self, low, high, inplace=False):
        """
        Crops the length (time) of the observations between the index low (inclusive) and between (exclusive). This
        operation does not invalidate windows because can be used as a helper function for other methods.
        :param low: Low index.
        :param high: Higher index.
        :param inplace: Make changes inplace.
        :return:
        """
        obs_set = self if inplace else self.copy()
        obs_set.obs = obs_set.obs[:,:,low:high]

        obs_set.length_vector = np.minimum(high - low, obs_set.length_vector - low)
        return obs_set

    def min_length(self):
        """
        Returns the length of the shortest observation.
        :return: Length (time) of the shortest observations
        """
        return self.length_vector.min()

    def select_observations(self, indices, inplace=False):
        """
        Returns the observations in the indices selected.
        :param indices: Indices to keep.
        :param inplace: If inplace=True deletes all observations but the observations on indices.
        :return: ObservationSet containing only the selected observations.
        """
        obs_set = self if inplace else self.copy()
        indicesar = np.asarray(indices)

        obs_set.obs = obs_set.obs[indicesar, :, :]
        obs_set.length_vector = obs_set.length_vector[indicesar]
        obs_set.names_vector = obs_set.names_vector[indicesar]

        if obs_set.obs.ndim == 2:
            obs_set.obs = obs_set.obs[np.newaxis,:,:]
            obs_set.length_vector = np.asarray([obs_set.length_vector])
            obs_set.names_vector = np.asarray([obs_set.names_vector])

        obs_set.crop_to_max_index(inplace=True)
        return obs_set

    def unselect_observations(self, indices, inplace=False):
        """
        Returns the observations that are not within the indices array.
        :param indices: Observation indices that will not be selected.
        :param inplace: If inplace=True, deletes the observations in indices.
        :return:
        """
        obs_set = self if inplace else self.copy()
        total_indices = np.arange(obs_set.num_observations())
        return obs_set.select_observations(np.delete(total_indices, indices), inplace=True)

    def search_names(self, glob_names):
        """
        Returns the indices where some names can be found in the observation set.
        :param glob_name:
        :return:
        """
        namesar = np.asarray(glob_names)
        selected_names = []

        for glob_name in namesar:
            matched_name = fnmatch.filter(self.names_vector, glob_name)
            selected_names.extend(matched_name)

        selected_indices = np.where(np.in1d(self.names_vector, selected_names))[0]

        return selected_indices

    def unselect_names(self, names, inplace=False):
        """
        Unselects observations by its name.
        :param names: Names to be unselected. Can be a single str or a list or a numpy array.
        :param inplace:
        :return:
        """

        obs_set = self if inplace else self.copy()
        unselect_indices = obs_set.search_names(names)
        return obs_set.unselect_observations(unselect_indices, inplace=True)

    def select_names(self, names, inplace=False):
        """
        Select observations by its name.
        :param names: Names to be selected. Can be a single str or a list or a numpy array.
        :param inplace:
        :return:
        """
        obs_set = self if inplace else self.copy()
        select_indices = obs_set.search_names(names)
        if select_indices.size == 0:
            obs_set.obs = np.asarray([])
            obs_set.length_vector = np.asarray([])
            obs_set.names_vector = np.asarray([])
            return obs_set
        return obs_set.select_observations(select_indices, inplace=True)

    def axis1d(self, axis):
        """
        Returns a 1-D array with the observations in the given axes, hiding the non-existent (-1) values.
        :param axis: Axis to keep.
        :return: 1-D array with the observations in axis.
        """
        ar_axis = np.asarray(axis)
        if ar_axis.ndim == 0:
            ar_axis = np.asarray([ar_axis])

        obs_1d = np.empty(self.length_vector.sum() * ar_axis.shape[0])
        length_cumsum = np.hstack((0,np.cumsum(self.length_vector))) * ar_axis.shape[0]
        for n in range(0, self.num_observations()):
            start_pos = length_cumsum[n]
            end_pos = length_cumsum[n+1]
            obs_1d[start_pos:end_pos] = self.obs[n,ar_axis,:self.length_vector[n]].reshape(-1)

        return obs_1d

    def _window_limits(self):
        """
        Return the frame number limit for each window. The very first windows can
        have 1 more frame because of the remaining frames.
        :param nWindow: Number of desired window.
        :return: ndarray where result[0] is the first frame of the second window.
        """
        n_frames_window = int(self.min_length() / self.n_window)

        remaining = self.min_length() - n_frames_window * self.n_window
        limits = np.empty((self.n_window,), dtype=np.int32)
        current_limit = 0
        for i in range(0, remaining):
            current_limit += n_frames_window + 1
            limits[i] = current_limit
        for i in range(remaining, self.n_window):
            current_limit += n_frames_window
            limits[i] = current_limit
        return limits

    def _update_window_limits(self):
        """
        Updates the window limits when there are some changes in the length of observations.
        :return: Updates window limits.
        """
        self.window_limits = self._window_limits()
        self.window_start = np.empty(self.window_limits.shape, self.window_limits.dtype)
        self.window_start[0] = 0
        self.window_start[1:] = self.window_limits[:-1]

    def _invalidate_window(self):
        """
        Invalidates the execution of temporal windows because of some change in the ObservationSet incompatible with
        the temporal window model.
        :return:
        """
        self.n_window = -1
        self.window_limits = None
        self.window_start = None

    def get_window_observation(self, index, inplace=False):
        """
        Gets an ObservationSet with the observations in the index-th temporal window.
        :param index: Index of temporal window.
        :param inplace: Make inplace changes.
        :return: Cropped ObservationSet.
        """
        cropped_window =  self._crop_between_indices(self.window_start[index], self.window_limits[index], inplace)
        cropped_window.n_window = 1
        cropped_window._update_window_limits()
        return cropped_window

    def get_selection(self, min_col, max_col, min_row, max_row):
        """
        Gets the indices of the workpieces and its temporal moments where the position is between min_col and max_col for
        the columns axis and between min_row and max_row for the rows axis.

        :param min_col: Minimum value (inclusive) for the columns axis.
        :param max_col: Maximum value (exclusive) for the columns axis.
        :param min_row: Minimum value (inclusive) for the rows axis.
        :param max_row: Maximum value (exclusive) for the rows axis.
        :return: Indices of the workpieces,
                Indices of the temporal moments.
        """
        axis_col_boolean = np.logical_and(self.obs[:,0,:] >= min_col, self.obs[:, 0, :] < max_col)
        axis_row_boolean = np.logical_and(self.obs[:,1,:] >= min_row, self.obs[:, 1, :] < max_row)
        index_workpiece, index_frame = np.where(np.logical_and(axis_col_boolean, axis_row_boolean))
        return index_workpiece, index_frame

    def get_valid_selection(self, min_col, max_col, min_row, max_row):
        """
        Gets the indices of the observations in the bounding box defined by [min_col, max_col) and [min_row, max_row). It only
        returns a valid selection, so observations in the bounds of the ObservationSet are not considered.
        :param min_col: Minimum value (inclusive) for the columns axis.
        :param max_col: Maximum value (exclusive) for the columns axis.
        :param min_row: Minimum value (inclusive) for the rows axis.
        :param max_row: Maximum value (exclusive) for the rows axis.
        :return: Indices of the workpieces,
                Indices of the temporal moments.
        """
        index_workpiece_origin, index_frame_origin = self.get_selection(min_col, max_col, min_row, max_row)
        index_frame_destination = index_frame_origin + 1
        length_workpiece = self.length_vector[index_workpiece_origin]
        wrong_index = np.where(index_frame_destination >= length_workpiece)[0]
        if wrong_index.size:
            index_workpiece_origin = np.delete(index_workpiece_origin, wrong_index)
            index_frame_origin = np.delete(index_frame_origin, wrong_index)
        return index_workpiece_origin, index_frame_origin

    def get_valid_window_array(self):
        """
        Gets the observations in a window in a 2-D array [2, N]. This method collapses all the observations in a single
        array, omitting all the observations out of the window.
        :return: 2-D array containing the observations for the current single-window ObservationSet
        """
        if self.n_window > 1:
            raise ValueError(
                "The ObservationSet contains more than 1 window. Select a window before calling get_valid_window_array()")

        return self.obs[:,:,self.window_start[0]:self.window_limits[0]].swapaxes(0,1).reshape(2,-1)

    def space_bounds(self):
        """
        Returns the space bounds for this ObservationSet. The first argument returned contains the columns axis [min max] value.
        Same format is applied for the rows axis in the second argument returned.
        :return: [min_col, max_col], [min_row, max_row] values.
        """
        min_col = np.inf
        max_col = -np.inf
        min_row = np.inf
        max_row = -np.inf
        for n in range(0,self.num_observations()):
            len_obs = self.length_vector[n]
            if self[n,0,:len_obs].min() < min_col:
                min_col = self[n,0,:len_obs].min()
            if self[n,0,:len_obs].max() > max_col:
                max_col = self[n,0,:len_obs].max()

            if self[n, 1, :len_obs].min() < min_row:
                min_row = self[n, 1, :len_obs].min()
            if self[n, 1, :len_obs].max() > max_row:
                max_row = self[n, 1, :len_obs].max()

        return np.asarray([min_col, max_col]), \
               np.asarray([min_row, max_row])

    def num_observations(self):
        return self.obs.shape[0]

    def num_windows(self):
        """
        Returns the number of defined temporal windows.
        :return: Number of temporal windows.
        """
        return self.n_window

    def copy(self):
        return ObservationSet(self.obs.copy(),
                              self.length_vector.copy(),
                              self.names_vector.copy(),
                              n_window=self.n_window)

class ObservationROISet(ObservationSet):
    """
    This class implements an ObservationSet with a ROI (region of interest applied). It implements the following
    additional attributes:

    - ini_obstacle[1/2]: 1-D arrays with the initial frame for the [first/second] obstacle. If there is no such obstacle, nan values
    are used.
    - end_obstacle[1/2]: 1-D arrays with the last frame for the [first/second] obstacle. If there is no such obstacle, nan values
    are used.
    - has_obstacle[1/2]: A boolean indicating if the ObservationROISet has the [first/second] obstacle.
    """
    def __init__(self, observations, length_vector, names_vector, ini_obstacle1, end_obstacle1, ini_obstacle2, end_obstacle2):
        super(ObservationROISet, self).__init__(observations, length_vector, names_vector, n_window=0)
        self.ini_obstacle1 = ini_obstacle1
        self.end_obstacle1 = end_obstacle1
        self.ini_obstacle2 = ini_obstacle2
        self.end_obstacle2 = end_obstacle2
        self.has_obstacle1 = False
        self.has_obstacle2 = False

        self.n_window = 1
        if not np.isnan(self.ini_obstacle1).sum() and not np.isnan(self.end_obstacle1).sum():
            self.has_obstacle1 = True
            self.n_window += 2
        if not np.isnan(self.ini_obstacle2).sum() and not np.isnan(self.end_obstacle2).sum():
            self.has_obstacle2 = True
            self.n_window += 2

        self.valid_start = np.full((self.num_observations(),), 0)
        self.valid_end = self.length_vector
        self._update_window_limits()

    @classmethod
    def fromfolder(self, data_folder):
        """
        Loads the data from a folder name. The data is a collection of npz files with the differentation videos and a
        metadata.pkl file with the metadata. From the differentiation videos, the laser spot positions are computed
        and an ObservationROISet is initialized.
        :param data_folder: Name of the folder containing the data.
        :return: ObservationROISet of the data.
        """
        with open(os.path.join(data_folder, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            max_lines = max([metadata[file]['SizeInFrames'] for file in metadata.keys()])

            n_files = len(metadata)

            observations = np.full((n_files, 2, max_lines), -1, dtype=np.float)
            length_vector = np.empty((n_files,), dtype=np.int)
            names_vector = np.empty((n_files,), dtype='object')
            ini_obstacle1 = np.full((n_files,), np.nan)
            end_obstacle1 = np.full((n_files,), np.nan)
            ini_obstacle2 = np.full((n_files,), np.nan)
            end_obstacle2 = np.full((n_files,), np.nan)

            # Reads each file.
            for n, file_key in enumerate(metadata.keys()):
                file_info = metadata[file_key]
                diff_video = np.load(data_folder + "/" + file_key + ".npz")['image']
                x, y = misc.weighted_gravity_centers(diff_video, threshold=0)
                x = misc.fix_nan(x)
                y = misc.fix_nan(y)

                observations[n, 0, 0:x.shape[0]] = x
                observations[n, 1, 0:y.shape[0]] = y
                length_vector[n] = file_info['SizeInFrames']
                names_vector[n] = os.path.basename(file_key)
                if file_info['KeyfIniObstacle1'] != -1 and file_info['KeyfEndObstacle1'] != -1:
                    ini_obstacle1[n] = file_info['KeyfIniObstacle1']
                    end_obstacle1[n] = file_info['KeyfEndObstacle1']
                if file_info['KeyfIniObstacle2'] != -1 and file_info['KeyfEndObstacle2'] != -1:
                    ini_obstacle2[n] = file_info['KeyfIniObstacle2']
                    end_obstacle2[n] = file_info['KeyfEndObstacle2']

            ObservationROISet._check_nan_obstacles(ini_obstacle1)
            ObservationROISet._check_nan_obstacles(end_obstacle1)
            ObservationROISet._check_nan_obstacles(ini_obstacle2)
            ObservationROISet._check_nan_obstacles(end_obstacle2)

            return ObservationROISet(observations, length_vector, names_vector, ini_obstacle1, end_obstacle1, ini_obstacle2, end_obstacle2)

    @classmethod
    def _check_nan_obstacles(cls, obstacle_frames):
        """
        Checks that the obstacle frames array have nan values on every position of the array or a concrete value for the
        obstacle frame. A mix of concrete values and nan values are not allowed.
        :param obstacle_frames: 1-D array containing information about the start and end of the obstacle.
        :return: A warning is returned if an error is found.
        """
        sum_nans = np.isnan(obstacle_frames).sum()
        if sum_nans > 0 and sum_nans < sum_nans.size:
            warnings.warn("Some obstacle frames has nan values while other obstacle frames have a correct value.")
            sys.exit("An error ocurred while processing obstacle info.")

    def crop_to_min_index(self, inplace=True):
        """
        Crops the observations to the length of the shortest observation. Destructive operation.
        :param inplace:
        :return:
        """
        obs_set = super(ObservationROISet, self).crop_to_min_index(inplace)
        obs_set.valid_end = obs_set.length_vector
        return obs_set

    def crop_between_indices(self, low, high, inplace=False):
        """
        Crops the length (time) of the observations between the index low (inclusive) and between (exclusive)
        :param low: Lower bound of the cropped range.
        :param high: Upper bound of the cropped range.
        :param inplace:
        :return:
        """
        obs_set = super(ObservationROISet, self).crop_between_indices(low, high, inplace)

        obs_set.ini_obstacle1 = ObservationROISet._update_obstacle_limits(low, high, obs_set.ini_obstacle1)
        obs_set.end_obstacle1 = ObservationROISet._update_obstacle_limits(low, high, obs_set.end_obstacle1)
        obs_set.ini_obstacle2 = ObservationROISet._update_obstacle_limits(low, high, obs_set.ini_obstacle2)
        obs_set.end_obstacle2 = ObservationROISet._update_obstacle_limits(low, high, obs_set.end_obstacle2)

        obs_set.valid_start[:] = 0
        obs_set.valid_end = obs_set.length_vector
        return obs_set

    @classmethod
    def _update_obstacle_limits(self, low, high, obstacle_array):
        """
        Update the values for the obstacle_array when the ObservationROISet is cropped. If the obstacle_array event is not
        within the cropped range, the array is set to np.nan
        :param low: Lower bound of the cropped range.
        :param high: Higher bound of the cropped range.
        :param obstacle_array: obstacle array to update to correspond to the cropped range.
        :return: Updated obstacle array.
        """
        # If obstacle position is higher than the upper bound, obstacle positions will happen after the cropped fragment [low:high)
        obstacle_array[obstacle_array > high] = np.nan
        # Updates the obstacle positions
        obstacle_array = obstacle_array - low
        # If obstacle position is negative, the obstacle positions is previous to the cropped fragment [low:high).
        obstacle_array[obstacle_array < 0] = np.nan

        return obstacle_array

    def select_observations(self, indices, inplace=False):
        """
        Returns the observations in the indices selected.
        :param indices: Indices to keep.
        :param inplace: If inplace=True deletes all observations but the observations on indices.
        :return: ObservationSet containing only the selected observations.
        """
        indicesar = np.asarray(indices)
        obs_set = super(ObservationROISet, self).select_observations(indicesar,inplace)


        obs_set.ini_obstacle1 = obs_set.ini_obstacle1[indicesar]
        obs_set.end_obstacle1 = obs_set.end_obstacle1[indicesar]
        obs_set.ini_obstacle2 = obs_set.ini_obstacle2[indicesar]
        obs_set.end_obstacle2 = obs_set.end_obstacle2[indicesar]
        obs_set.valid_start = obs_set.valid_start[indicesar]
        obs_set.valid_end = obs_set.valid_end[indicesar]

        if obs_set.ini_obstacle1.ndim == 0:
            obs_set.ini_obstacle1 = np.asarray([obs_set.ini_obstacle1])
            obs_set.end_obstacle1 = np.asarray([obs_set.end_obstacle1])
            obs_set.ini_obstacle2 = np.asarray([obs_set.ini_obstacle2])
            obs_set.end_obstacle2 = np.asarray([obs_set.end_obstacle2])
            obs_set.valid_start = np.asarray([obs_set.valid_start])
            obs_set.valid_end = np.asarray([obs_set.valid_end])


        obs_set._update_window_limits()
        return obs_set

    def get_window_observation(self, index, inplace=False):
        """
        Gets an ObservationSet with the observations in the index-th temporal window.
        :param index: Index of temporal window.
        :param inplace: Make inplace changes.
        :return:
        """

        cropped_window = self._crop_between_indices(self.window_start[index], self.window_limits[index], inplace)
        cropped_window.valid_start, cropped_window.valid_end = self._new_valid_values_window(index)
        cropped_window.n_window = 1
        return cropped_window

    def _new_valid_values_window(self, index):
        """
        Returns the indices of the valid indices for the index-th window for the observation set. The indices are
        returned as two different 1-D arrays: the first array for the valid start indices and the other for the
        valid end indices. Each array has self.num_observations() elements.
        :param index: Index of temporal window.
        :return: 1-D array containing the valid start indices, 1-D array containing the valid end indices.
        """
        if self.n_window == 1:
            return np.full((self.num_observations(),),0), self.length_vector

        if index < 0:
            index = self.n_window + index

        if self.has_obstacle1 and not self.has_obstacle2:
            if index == 0:
                return np.full((self.num_observations(),),0), self.ini_obstacle1.astype(np.int)
            elif index == 1:
                return (self.ini_obstacle1 - self.ini_obstacle1.min()).astype(np.int), (self.end_obstacle1 - self.ini_obstacle1.min()).astype(np.int)
            else:
                return (self.end_obstacle1 - self.end_obstacle1.min()).astype(np.int), (self.length_vector - self.end_obstacle1.min()).astype(np.int)
        elif not self.has_obstacle1 and self.has_obstacle2:
            if index == 0:
                return np.full((self.num_observations(),), 0), self.ini_obstacle2.astype(np.int)
            elif index == 1:
                return (self.ini_obstacle2 - self.ini_obstacle2.min()).astype(np.int), (self.end_obstacle2 - self.ini_obstacle2.min()).astype(np.int)
            else:
                return (self.end_obstacle2 - self.end_obstacle2.min()).astype(np.int), (self.length_vector - self.end_obstacle2.min()).astype(np.int)
        elif self.has_obstacle1 and self.has_obstacle2:
            if index == 0:
                return np.full((self.num_observations(),), 0), self.ini_obstacle1.astype(np.int)
            elif index == 1:
                return (self.ini_obstacle1 - self.ini_obstacle1.min()).astype(np.int), (self.end_obstacle1 - self.ini_obstacle1.min()).astype(np.int)
            elif index == 2:
                return (self.end_obstacle1 - self.end_obstacle1.min()).astype(np.int), (self.ini_obstacle2 - self.end_obstacle1.min()).astype(np.int)
            elif index == 3:
                return (self.ini_obstacle2 - self.ini_obstacle2.min()).astype(np.int), (self.end_obstacle2 - self.ini_obstacle2.min()).astype(np.int)
            else:
                return (self.end_obstacle2 - self.end_obstacle2.min()).astype(np.int), (self.length_vector - self.end_obstacle2.min()).astype(np.int)
        else:
            warnings.warn("Some error found during the selection of window in ObservationROISet.")

    def _update_window_limits(self):
        """
        Update the window limits when there are some changes in the ObservationSet.
        :return:
        """
        self.window_limits = np.empty((self.n_window), dtype=np.int32)
        self.window_start = np.empty(self.window_limits.shape, self.window_limits.dtype)
        self.window_start[0] = 0

        curr_ix = 0
        if self.has_obstacle1:
            self.window_limits[curr_ix] = self.ini_obstacle1.max()
            curr_ix += 1
            self.window_limits[curr_ix] = self.end_obstacle1.max()
            self.window_start[curr_ix] = self.ini_obstacle1.min()
            curr_ix += 1
            self.window_start[curr_ix] = self.end_obstacle1.min()

        if self.has_obstacle2:
            self.window_limits[curr_ix] = self.ini_obstacle2.max()
            curr_ix += 1
            self.window_limits[curr_ix] = self.end_obstacle2.max()
            self.window_start[curr_ix] = self.ini_obstacle2.min()
            curr_ix += 1
            self.window_start[curr_ix] = self.end_obstacle2.min()

        self.window_limits[curr_ix] = self.length_vector.max()

    def get_valid_selection(self, min_col, max_col, min_row, max_row):
        """
        Gets the indices of the observations in the bounding box defined by [min_col, max_col) and [min_row, max_row). It only
        returns a valid selection, so observations in the bounds of the ObservationSet are not considered.
        :param min_col: Minimum value (inclusive) for the columns axis.
        :param max_col: Maximum value (exclusive) for the columns axis.
        :param min_row: Minimum value (inclusive) for the rows axis.
        :param max_row: Maximum value (exclusive) for the rows axis.
        :return: Indices of the workpieces,
                Indices of the temporal moments.
        """
        index_workpiece_origin, index_frame_origin = self.get_selection(min_col, max_col, min_row, max_row)
        index_frame_destination = index_frame_origin + 1
        workpiece_valid_start = self.valid_start[index_workpiece_origin]
        workpiece_valid_end = self.valid_end[index_workpiece_origin]

        between_wrong_indices = np.logical_or(index_frame_destination < workpiece_valid_start, index_frame_destination >= workpiece_valid_end)
        wrong_index = np.where(between_wrong_indices)[0]

        if wrong_index.size:
            index_workpiece_origin = np.delete(index_workpiece_origin, wrong_index)
            index_frame_origin = np.delete(index_frame_origin, wrong_index)
        return index_workpiece_origin, index_frame_origin

    def get_valid_window_array(self):
        """
        Gets the observations in a window in a 2-D array [2, N]. This method collapses all the observations in a single array, omitting
        all the observations out of the window.
        :return: 2-D array containing the observations for the current single-window ObservationSet
        """
        if self.n_window > 1:
            raise ValueError("The ObservationSet contains more than 1 window. Select a window before calling get_valid_window_array()")

        n_elements = (self.valid_end - self.valid_start).sum()

        array = np.empty((2,n_elements))

        curr_start = 0
        for i in range(0,self.num_observations()):
            window_length = self.valid_end[i] - self.valid_start[i]
            array[:,curr_start:curr_start+window_length] = self.obs[i,:,self.valid_start[i]:self.valid_end[i]]
            curr_start += window_length

        return array

    def __getitem__(self, item):
        return self.obs[item]

    def __repr__(self):
        return "ObservationROISet with " + str(self.num_observations()) + "x" + str(self.obs.shape[2]) +  " obs. [" + str(self.num_windows()) + " window]"

    def copy(self):
        return ObservationROISet(self.obs.copy(),
                                 self.length_vector.copy(),
                                 self.names_vector.copy(),
                                 self.ini_obstacle1.copy(),
                                 self.end_obstacle1.copy(),
                                 self.ini_obstacle2.copy(),
                                 self.end_obstacle2.copy())

class SymbolizationType(Enum):
    """
    This enum defines the type of symbolization used. The difference between EQUAL_FREQUENCY and EQUAL_FREQUENCY_NO_BOUNDS
    is the way it treats the bounds of the space. In the EQUAL_FREQUENCY, the region division covers all the space
    [0,32] x [0,32]. In the EQUAL_FREQUENCY_NO_BOUNDS, the division of region could not cover all the space, and instead
    covers the [xMin, xMax] [yMin, yMax], where xMin, xMax, yMin and yMax are the lowest and greatest x and y coordinates
    found in the training set.

    Take into account that EQUAL_FREQUENCY and EQUAL_FREQUENCY_NO_BOUNDS produce different results depending on which
    axis is divided first. See also :class:`DivisionOrder`.
    """
    EQUAL_WIDTH = 1
    EQUAL_FREQUENCY = 2
    EQUAL_FREQUENCY_NO_BOUNDS = 3

class DivisionOrder(Enum):
    """
    This enum decide how the symbolization will be executed when :attr:`SymbolizationType.EQUAL_FREQUENCY` and
    :attr:`SymbolizationType.EQUAL_FREQUENCY_NO_BOUNDS` is specified. If ROWS_THEN_COLUMNS, the space is first divided in rows,
    and then each row is divided into different columns. Same corresponding behaviour is used for COLUMNS_THEN_ROWS.

    Take into account that the limits of the second division (columns in ROWS_THEN_COLUMNS, rows in COLUMNS_THEN_ROWS),
    are different for each slice of the first division (rows in ROWS_THEN_COLUMNS, columns in COLUMNS_THEN_ROWS).
    """
    ROWS_THEN_COLUMNS = 1
    COLUMNS_THEN_ROWS = 2

class EqualWidthLimits(object):
    """
    Divides the space by EQUAL_WIDTH. Refer to :class:`SymbolizationType`.
    """
    def __init__(self, train_obs_set, n_cell_col, n_cell_row):
        col_bounds, row_bounds = train_obs_set.space_bounds()
        self.col_limits = np.linspace(0, col_bounds[1], n_cell_col)
        self.row_limits = np.linspace(0, row_bounds[1], n_cell_row)
        self.n_cell_col = n_cell_col
        self.n_cell_row = n_cell_row

    def symbolizate(self, test_obs_set):
        """
        Converts a observation set to an array of symbols.
        :param test_obs_set: ObservationSet.
        :return: 2-D array of symbols.
        """
        row_axis_symb = (np.digitize(test_obs_set[:,1,:], self.row_limits) - 1)
        row_out_bounds = np.logical_or(row_axis_symb < 0, row_axis_symb >= self.n_cell_row)

        col_axis_symb = (np.digitize(test_obs_set[:,0,:], self.col_limits) - 1)
        col_out_bounds = np.logical_or(col_axis_symb < 0, col_axis_symb >= self.n_cell_col)

        symbols = row_axis_symb * self.n_cell_col + col_axis_symb

        out_bounds = np.logical_or(row_out_bounds, col_out_bounds)
        symbols[out_bounds] = -1
        return symbols

class EqualFrequencyLimits(object):
    """
    Divides the space by EQUAL_FREQUENCY. Refer to :class:`SymbolizationType` and :class:`DivisionOrder`.
    """
    def __init__(self, train_obs_set, n_cell_col, n_cell_row, division_order):
        self.division_order = division_order

        if division_order == DivisionOrder.ROWS_THEN_COLUMNS:
            self.first_idx = 1
            self.second_idx = 0
            self.first_ncell = n_cell_row
            self.second_ncell = n_cell_col
        elif division_order == DivisionOrder.COLUMNS_THEN_ROWS:
            self.first_idx = 0
            self.second_idx = 1
            self.first_ncell = n_cell_col
            self.second_ncell = n_cell_row
        else:
            raise TypeError("A valid DivisionOrder should be used.")

        self._generate_limits(train_obs_set)

    def _generate_limits(self, train_obs_set):
        """
        Divides the space using EQUAL_FREQUENCY. This function generates :attr:`self.first_limits`  and
        :attr:`self.second_limits` to divide the space.
        :param train_obs_set: Training ObservationSet.
        :return:
        """
        self.first_limits = self._axis_equal_frequency_limits(train_obs_set.axis1d(self.first_idx), self.first_ncell, 0, 32.01)
        self.second_limits = np.empty((self.first_ncell, self.second_ncell+1))

        for axis in range(0, self.first_ncell):
            if self.division_order == DivisionOrder.ROWS_THEN_COLUMNS:
                min_row = self.first_limits[axis]
                max_row = self.first_limits[axis + 1]
                workpiece_in_axis, frame_in_axis = train_obs_set.get_selection(0, 32, min_row, max_row)
            elif self.division_order == DivisionOrder.COLUMNS_THEN_ROWS:
                min_col = self.first_limits[axis]
                max_col = self.first_limits[axis + 1]
                workpiece_in_axis, frame_in_axis = train_obs_set.get_selection(min_col, max_col, 0, 32)
            else:
                raise TypeError("A valid DivisionOrder should be used.")

            obs_in_axis = train_obs_set[workpiece_in_axis, self.second_idx, frame_in_axis].reshape(-1)
            self.second_limits[axis] = self._axis_equal_frequency_limits(obs_in_axis, self.second_ncell, 0, 32.01)

    def _axis_equal_frequency_limits(self, axis_obs, n_cell, min_value, max_value):
        """
        Divides the observations in the 1-D array axis_obs using equal_frequency. min_value and max_value indicates the
        space bounds for the space division.
        :param axis_obs: 1-D array containing observations.
        :param n_cell: Number of cells to divide on.
        :param min_value: Lower bound for the axis.
        :param max_value: Upper bound for the axis.
        :return: A 1-D array with n_cell+1 values, where (counting from 0) the i-th cell starts at the i index and ends in the
        i+1-th index.
        """
        sorted_obs = np.sort(axis_obs)
        obs_per_cell = int(sorted_obs.shape[0] / n_cell)
        remaining_obs = sorted_obs.shape[0] - obs_per_cell * n_cell

        division_indices = np.linspace(obs_per_cell, obs_per_cell * n_cell, n_cell, dtype=np.int)[:-1]
        benefited_indices = random.sample(range(division_indices.shape[0]), remaining_obs)
        for idx in benefited_indices:
            division_indices[idx:] += 1

        axis_limits = np.empty((n_cell + 1,))
        axis_limits[0] = min_value
        axis_limits[-1] = max_value

        axis_limits[1:-1] = sorted_obs[division_indices]

        return axis_limits

    def symbolizate(self, test_obs_set):
        """
        Converts a observation set to an array of symbols.
        :param test_obs_set: ObservationSet.
        :return: 2-D array of symbols.
        """
        first_axis = (np.digitize(test_obs_set[:,self.first_idx,:], self.first_limits) - 1)
        first_out_bounds = np.logical_or(first_axis < 0, first_axis >= self.first_ncell)

        second_axis = np.full(first_axis.shape, -1,  dtype=np.int)
        for axis in range(0, self.first_ncell):
            workpiece_in_axis, frames_in_axis = np.where(first_axis == axis)
            second_axis[workpiece_in_axis,frames_in_axis] = \
                np.digitize(test_obs_set[workpiece_in_axis,self.second_idx,frames_in_axis], self.second_limits[axis]) - 1

        second_out_bounds = np.logical_or(second_axis < 0, second_axis >= self.second_ncell)

        symbols = second_axis * self.first_ncell + first_axis
        out_bounds = np.logical_or(first_out_bounds, second_out_bounds)
        symbols[out_bounds] = -1
        return symbols

class EqualFrequencyLimitsNoBounds(EqualFrequencyLimits):
    """
    Divides the space by EQUAL_FREQUENCY_NO_BOUNDS. Refer to :class:`SymbolizationType`.
    """
    def __init__(self, train_obs_set, n_cell_col, n_cell_row, division_order):
        super(EqualFrequencyLimitsNoBounds, self).__init__(train_obs_set, n_cell_col, n_cell_row, division_order)

    def _generate_limits(self, train_obs_set):
        """
        Divides the space using EQUAL_FREQUENCY_NO_BOUNDS. This function generates :attr:`self.first_limits`  and
        :attr:`self.second_limits` to divide the space. The lower and upper bounds of each space axis is obtained from
        the train ObservationSet.
        :param train_obs_set: Training ObservationSet.
        :return:
        """
        first_axis_obs = train_obs_set.axis1d(self.first_idx)
        first_axis_min = max(0, first_axis_obs.min() - 0.01)
        first_axis_max = min(32, first_axis_obs.max() + 0.01)
        self.first_limits = self._axis_equal_frequency_limits(first_axis_obs,
                                                              self.first_ncell, first_axis_min,
                                                              first_axis_max)
        self.second_limits = np.empty((self.first_ncell, self.second_ncell+1))

        for axis in range(0, self.first_ncell):
            if self.division_order == DivisionOrder.ROWS_THEN_COLUMNS:
                min_row = self.first_limits[axis]
                max_row = self.first_limits[axis + 1]
                workpiece_in_axis, frame_in_axis = train_obs_set.get_selection(0, 32, min_row, max_row)
            elif self.division_order == DivisionOrder.COLUMNS_THEN_ROWS:
                min_col = self.first_limits[axis]
                max_col = self.first_limits[axis + 1]
                workpiece_in_axis, frame_in_axis = train_obs_set.get_selection(min_col, max_col, 0, 32)
            else:
                raise TypeError("A valid DivisionOrder should be used.")

            obs_in_axis = train_obs_set[workpiece_in_axis, self.second_idx, frame_in_axis].reshape(-1)

            second_axis_min = max(0, obs_in_axis.min()-0.01)
            second_axis_max = min(32, obs_in_axis.max()+0.01)

            self.second_limits[axis] = self._axis_equal_frequency_limits(obs_in_axis, self.second_ncell,
                                                                         second_axis_min, second_axis_max)

class SymbolSet(object):
    """
    This class defines the equivalent abstraction of ObservationSet for the symbolization in the D-Markov machines.

    The implementation is pretty similar to ObservationSet. It includes the following attributes.
    - symbols: as the obs attribute in the ObservationSet, it represents the symbol of each temporal moment of each
        video in a 2-D array (n_videos x max_length_video).
    - length_vector: As in ObservationSet.
    - names_vector: As in ObservationSet.

    """
    def __init__(self, symbols, length_vector, names_vector):
        self.symbols = symbols
        self.length_vector = length_vector
        self.names_vector = names_vector

    @classmethod
    def from_observationset(cls, obs_set, n_cell_col, n_cell_row, symbolization_type=SymbolizationType.EQUAL_WIDTH,
                            division_order=DivisionOrder.ROWS_THEN_COLUMNS):
        """
        Generates a SymbolSet from an ObservationSet and a symbolization type/division order.
        :param obs_set: ObservationSet to translate.
        :param n_cell_col: Number of divisions in the columns axis.
        :param n_cell_row: Number of divisions in the rows axis.
        :param symbolization_type: Symbolization type of type SymbolizationType.
        :param division_order: Division order of type DivisionOrder.
        :return:
        """
        if symbolization_type == SymbolizationType.EQUAL_WIDTH:
            symbolizer = EqualWidthLimits(obs_set, n_cell_col, n_cell_row)
        elif symbolization_type == SymbolizationType.EQUAL_FREQUENCY:
            symbolizer = EqualFrequencyLimits(obs_set, n_cell_col, n_cell_row, division_order)
        elif symbolization_type == SymbolizationType.EQUAL_FREQUENCY_NO_BOUNDS:
            symbolizer = EqualFrequencyLimitsNoBounds(obs_set, n_cell_col, n_cell_row, division_order)
        else:
            raise TypeError("A valid SymbolizationType should be used.")

        symbols = symbolizer.symbolizate(obs_set)
        return SymbolSet(symbols, obs_set.length_vector, obs_set.names_vector), symbolizer

    def __getitem__(self, item):
        return self.symbols[item]

    def __repr__(self):
        return "Symbols with " + str(self.num_symbolizations()) + "x" + str(self.symbols.shape[1]) + " symbols."

    def select_obs_symbols(self, indices, inplace=False):
        """
        Returns the symbolizations in the indices selected.
        :param indices: Indices to keep.
        :param inplace: If inplace=True deletes all symbolizations but the symbolizations on indices.
        :return: SymbolSet containing only the selected symbolizations.
        """
        symb_set = self if inplace else self.copy()
        indicesar = np.asarray(indices)

        symb_set.symbols = symb_set.symbols[indicesar,:,:]
        symb_set.length_vector = symb_set.length_vector[indicesar]
        symb_set.names_vector = symb_set.names_vector[indicesar]

        if symb_set.symbols.ndim == 1:
            symb_set.symbols = symb_set.symbols[np.newaxis,:,:]
            symb_set.length_vector = np.asarray([symb_set.length_vector])
            symb_set.names_vector = np.asarray([symb_set.names_vector])

        symb_set.crop_to_max_index(inplace=True)
        return symb_set

    def unselect_obs_symbols(self, indices, inplace=False):
        """
        Returns the symbolizations that are not within the indices array.
        :param indices: Symbolization indices that will not be selected.
        :param inplace: If inplace=True, deletes the symbolizations in indices.
        :return: SymbolSet without the selected symbolizations.
        """
        symb_set = self if inplace else self.copy()
        total_indices = np.arange(symb_set.num_symbolizations())
        return symb_set.select_obs_symbols(np.delete(total_indices, indices), inplace=True)

    def search_names(self, glob_names):
        """
        Returns the indices where some names can be found in the SymbolSet.
        :param glob_name:
        :return:
        """
        namesar = np.asarray(glob_names)
        selected_names = []

        for glob_name in namesar:
            matched_name = fnmatch.filter(self.names_vector, glob_name)
            selected_names.extend(matched_name)

        selected_indices = np.where(np.in1d(self.names_vector, selected_names))[0]

        return selected_indices

    def unselect_names(self, names, inplace=False):
        """
        Unselects symbolization by its name.
        :param names: Names to be unselected. Can be a single str or a list or a numpy array.
        :param inplace:
        :return:
        """

        symb_set = self if inplace else self.copy()
        unselect_indices = symb_set.search_names(names)
        return symb_set.unselect_obs_symbols(unselect_indices, inplace=True)

    def select_names(self, names, inplace=False):
        """
        Select symbolizations by its name.
        :param names: Names to be selected. Can be a single str or a list or a numpy array.
        :param inplace:
        :return:
        """
        symb_set = self if inplace else self.copy()
        select_indices = symb_set.search_names(names)
        if select_indices.size == 0:
            symb_set.obs = np.asarray([])
            symb_set.length_vector = np.asarray([])
            symb_set.names_vector = np.asarray([])
            return symb_set
        return symb_set.select_obs_symbols(select_indices, inplace=True)

    def crop_to_max_index(self, inplace=True):
        """
        Crops the symbols to the length of the current largest symbolization. This is usually useful when making changes
        to the SymbolSet, such as removing symbolizations, because the current symbolizations could be shorter than
        the largest original symbolization.
        :param inplace:
        :return:
        """
        symb_set = self if inplace else self.copy()
        max_index = symb_set.length_vector.max()

        if symb_set.symbols.shape[1] > max_index:
            symb_set.symbols = symb_set.symbols[:, :, :max_index]

        return symb_set

    def min_index(self):
        """
        Returns the length of the shortest symbolization.
        :return: Length (time) of the shortest symbolization
        """
        return self.length_vector.min()

    def num_symbolizations(self):
        return self.symbols.shape[0]

    def copy(self):
        return SymbolSet(self.symbols.copy(),
                         self.length_vector.copy(),
                         self.names_vector.copy())