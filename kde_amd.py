"""
This file implements the Kernel Density Estimation - Anomaly Movement Detector (KDE-AMD).
"""

import numpy as np
import scipy.stats as stats

import observations_set
import os

class KDEAMD(object):
    """
    KDE-AMD classifier.
    """
    def __init__(self, obs_train, min_unique_points, size_col=32, size_row=32, n_cell_col=16, n_cell_row=16):
        """
        Trains a KDE-AMD classifier using the ObservationSet obs_train. It trains a KDE model for each
        of the n_cell_col * n_cell_row regions. If some region does not contain at least min_unique_points, the region
        is left as empty.

        :param obs_train: ObservationSet or ObservationROISet to train the KDE-AMD.
        :param min_unique_points: Minimum number of points for each region of the KDE-AMD.
        :param size_col: Size of the domain in the columns axis.
        :param size_row: Size of the domain in the rows axis.
        :param n_cell_col: Number of regions in the columns axis.
        :param n_cell_row: Number of regions in the rows axis.
        """
        self.size_col = size_col
        self.size_row = size_row
        self.n_cell_col = n_cell_col
        self.n_cell_row = n_cell_row
        # Creates the space limits for each region. Index 0 is the start of the first cell and index 1 is the
        # end of the first cell. From there, index 2 is the end of the second cell and so on.
        self.col_limits = np.linspace(0, self.size_col, self.n_cell_col + 1)
        self.row_limits = np.linspace(0, self.size_row, self.n_cell_row + 1)
        # Temporal windows limits.
        self.window_start = obs_train.window_start.copy()
        self.window_limits = obs_train.window_limits.copy()
        self.kde_array = self._get_temporal_kde_array(obs_train, min_unique_points)

    def _get_temporal_kde_array(self, obs_train, min_unique_points):
        """
        Trains a grid of KDE models for each temporal window.
        :param obs_train: ObservationSet or ObservationROISet to train the KDE-AMD.
        :param min_unique_points: Minimum number of points for each region of the KDE-AMD.
        :return: KDE-AMD model.
        """
        kde_array = np.empty((obs_train.num_windows(), self.n_cell_row, self.n_cell_col), dtype=object)

        if obs_train.num_windows() > 1:
            kde_array[0] = self._get_kde_array_window(obs_train.get_window_observation(0), min_unique_points)
            for i in range(1, obs_train.num_windows()-1):
                kde_array[i] = self._get_kde_array_window(obs_train.get_window_observation(i), min_unique_points)
            kde_array[-1] = self._get_kde_array_window(obs_train.get_window_observation(-1), min_unique_points)
        else:
            kde_array[0] = self._get_kde_array_window(obs_train, min_unique_points)

        return kde_array

    def _get_kde_array_window(self, obs_train, min_unique_points):
        """
        Trains a KDE-AMD temporal window slice.
        :param obs_train: ObservationSet or ObservationROISet to train the KDE-AMD with observations
                of a temporal window.
        :param min_unique_points: Minimum number of points for each region of the KDE-AMD.
        :return: KDE-AMD 2-D matrix for a temporal window.
        """
        kde_array_slice = np.empty((self.n_cell_row, self.n_cell_col), dtype=object)

        for i in range(0, self.n_cell_row):
            for j in range(0, self.n_cell_col):
                index_workpiece_origin, index_frame_origin = obs_train.get_valid_selection(self.col_limits[j], self.col_limits[j + 1], self.row_limits[i], self.row_limits[i + 1])
                index_frame_destination = index_frame_origin + 1
                if index_frame_destination.size > min_unique_points:
                    partial_observations = obs_train.obs[index_workpiece_origin, :, index_frame_destination]
                    try:
                        kde_array_slice[i, j] = stats.kde.gaussian_kde(partial_observations.T)
                    except np.linalg.linalg.LinAlgError:
                        print("LinAlgError: Creation of KDE model. Indices: (" + str(i) + ", " + str(j) + ").")
        return kde_array_slice

    def evaluate_observation(self, obs_test):
        """
        Returns the probability of each movement in the ObservationSet or ObservationROISet. It applies a
        Laplace smoothing on the probabilities.
        :param obs_test: ObservationSet or ObservationROISet to classify.
        :return: 1-D numpy array of probabilities for each movement.
        """
        probabilities = self._observation_probability(obs_test)
        probabilities[probabilities == -1] = 0
        # Laplace smoothing
        probabilities += (0.000001 + probabilities) / 1.000001
        return probabilities

    def _observation_probability(self, obs_test):
        """
        Evaluates the KDE-AMD on the test data obs_set to return the probability of each movement.
        :param obs_test: ObservationSet or ObservationROISet to classify.
        :return: 1-D numpy array of probabilities for each movement.
        """
        if type(obs_test) is observations_set.ObservationSet:
            min_index = min(self.window_limits[-1], obs_test.min_length())
            probabilities = np.full((min_index-1,), -1, dtype=np.float)
            last_window_idx = np.argmax((min_index - 1) < self.window_limits)
            win_start = self.window_start
            win_limits = np.minimum(self.window_limits, min_index)
        elif type(obs_test) is observations_set.ObservationROISet:
            last_window_idx = obs_test.n_window - 1
            probabilities = np.full((obs_test.length_vector.max()-1,), -1, dtype=np.float)
            win_start = obs_test.window_start
            win_limits = obs_test.window_limits
        else:
            raise TypeError("obs_set is not from the correct class.")

        for n_window in range(0, last_window_idx-1):
            window_start_frame = win_start[n_window]
            window_limit_frame = win_limits[n_window]
            # Find region indices.
            cell_col = np.digitize(obs_test[0,0,window_start_frame:window_limit_frame], self.col_limits) - 1
            cell_row = np.digitize(obs_test[0,1,window_start_frame:window_limit_frame], self.row_limits) - 1
            array_kdes = self.kde_array[n_window, cell_col, cell_row]
            idx_non_none = np.where(array_kdes != None)[0]
            # Evaluate the KDE only for those positions where there is a trained KDE.
            for valid_idx in idx_non_none:
                probabilities[window_start_frame + valid_idx] = \
                    array_kdes[valid_idx](obs_test[0,:,window_start_frame + valid_idx + 1])

        # This code address the last window, where the last frame should not be taken into account because there is no
        # more frames to compute the movement
        window_start_frame = win_start[last_window_idx]
        window_limit_frame = win_limits[last_window_idx]
        cell_col = np.digitize(obs_test[0,0,window_start_frame:(window_limit_frame-1)], self.col_limits) - 1
        cell_row = np.digitize(obs_test[0,1,window_start_frame:(window_limit_frame-1)], self.row_limits) - 1
        array_kdes = self.kde_array[last_window_idx, cell_col, cell_row]
        idx_non_none = np.where(array_kdes != None)[0]

        for valid_idx in idx_non_none:
            probabilities[window_start_frame + valid_idx] = \
                array_kdes[valid_idx](obs_test[0,:,window_start_frame + valid_idx + 1])

        return probabilities

def evaluate_model(n_cell_col, n_cell_row, min_unique_points, data_folder, output_name):
    """
    Applies leaving-one-out training a KDE-AMD with n_cell_col divisions on the columns axis and
    n_cell_row divisions on the rows axis (See the KDEAMD class, also for the min_unique_points parameter).

    folder is the folder name of the data. output_name is the name of the file where the results will be saved.
    :param n_cell_col: Number of regions in the columns axis.
    :param n_cell_row: Number of regions in the rows axis.
    :param min_unique_points: Minimum number of points for each region of the KDE-AMD.
    :param data_folder: Folder name where the data is located.
    :param output_name: Name of the files which contains the result of the KDE-AMD using leaving-one-out.
    :return:
    """
    # Load data
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)

    obs_set.synchronize_average()

    dist_array = np.empty((obs_set.num_observations(),), dtype=object)
    gaussian_dist_array = np.empty((obs_set.num_observations(),), dtype=object)

    xmin = np.inf
    xmax = -np.inf

    for n in range(0,obs_set.num_observations()):
        print(str(n) + " / " + str(obs_set.num_observations()) + " " +  obs_set.names_vector[n])
        observations_train = obs_set.unselect_observations(n, inplace=False)
        observations_test = obs_set.select_observations(n, inplace=False)
        # Generate noise in the test data.
        observations_gaussian = observations_test.gaussian_noise(std_col=0.141421356, std_row=0.141421356, inplace=False)

        kde_amd = KDEAMD(observations_train, min_unique_points, n_cell_col=n_cell_col, n_cell_row=n_cell_row)

        probabilities = kde_amd.evaluate_observation(observations_test)
        probabilities_gaussian = kde_amd.evaluate_observation(observations_gaussian)

        log_likelihood = -np.log(probabilities)
        gaussian_loglikelihood = -np.log(probabilities_gaussian)

        dist_array[n] = stats.gaussian_kde(log_likelihood)
        gaussian_dist_array[n] = stats.gaussian_kde(gaussian_loglikelihood)

        if xmin > log_likelihood.min():
            xmin = log_likelihood.min()
        if xmin > gaussian_loglikelihood.min():
            xmin = gaussian_loglikelihood.min()
        if xmax < log_likelihood.max():
            xmax = log_likelihood.max()
        if xmax < gaussian_loglikelihood.max():
            xmax = gaussian_loglikelihood.max()

    # Generates the distribution of the loglikelihood.
    range_loglikelihood = np.linspace(xmin,xmax,2000)
    distribution_array = np.empty((obs_set.num_observations(), 2000))
    gaussian_distribution_array = np.empty((obs_set.num_observations(), 2000))

    # Evaluates the distribution of the loglikelihood for the normal/gaussian noise data.
    for n in range(0,obs_set.num_observations()):
        distribution_array[n] = dist_array[n](range_loglikelihood)
        gaussian_distribution_array[n] = gaussian_dist_array[n](range_loglikelihood)

    with open(output_name + '_normal.csv', 'w') as normal_file, open(output_name + '_gaussian002.csv', 'w') as gaussian_file:
        normal_file.write("Name,AnomalyScore" + '\n')
        gaussian_file.write("Name,AnomalyScore" + '\n')
        for n in range(0,obs_set.num_observations()):
            distributions_train = np.delete(distribution_array,n,axis=0)
            # The "correct" distribution of the loglikelihood is considered to be the mean of the distributions.
            mean_train = distributions_train.mean(axis=0)

            # Compute the Kullback-Leibler with respect the "correct" distribution of the loglikelihood.
            mean_kl_value = stats.entropy(mean_train, distribution_array[n])
            gaussian_mean_kl_value = stats.entropy(mean_train, gaussian_distribution_array[n])

            # Writes the anomaly score in the output files.
            print("Writing " + obs_set.names_vector[n])
            normal_file.write(obs_set.names_vector[n] + ',' + str(mean_kl_value) + '\n')
            gaussian_file.write(obs_set.names_vector[n] + ',' + str(gaussian_mean_kl_value) + '\n')

if __name__ == '__main__':
    windows = 35
    min_instances = 200

    if not os.path.isdir('results/KDEAMD'):
        os.mkdir('results/KDEAMD')

    print("Executing with " + str(windows) + "x" + str(windows) + " and lambda = " + str(min_instances))
    for workpiece_type in range(33, 37):
        data_folder = "data/Type" + str(workpiece_type)

        result_folder = "results/KDEAMD/Type" + str(workpiece_type)
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

        evaluate_model(windows, windows, min_instances, data_folder, result_folder + "/KDEAMD_" +
                       str(windows) + "_" + str(windows) + "_" + str(min_instances))