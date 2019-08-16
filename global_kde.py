"""
This file implements an anomaly detector using global KernelDensityEstimation models.
"""
import numpy as np
import scipy.stats as stats

import observations_set
import os

class GlobalKDE(object):
    """
    GlobalKDE implements a global Kernel Density Model for each temporal window.
    """
    def __init__(self, train_obs_set, grid_col=128, grid_row=128):
        """
        Implement a GlobalKDE model which generates a KDE model for each temporal window.
        :param train_obs_train: ObservationSet or ObservationROISet to train the GlobalKDE.
        :param grid_col: Number of elements in the columns axis in the grid to evaluate the KDE models.
        :param grid_row: Number of elements in the rows axis in the grid to evaluate the KDE models.
        """
        # Generates the grids where the KDE models are evaluated. Then, this grids are trained with the training data.
        self.train_grid = np.empty((train_obs_set.num_windows(), grid_col * grid_row))
        self._init_grid(grid_col, grid_row)

        # Train the grid for each temporal window.
        for n_window in range(0,train_obs_set.num_windows()):
            obs_window = train_obs_set.get_window_observation(n_window)
            train_array = obs_window.get_valid_window_array()
            kde_model = stats.gaussian_kde(train_array)
            self.train_grid[n_window] = kde_model(self.grid_coords.T)

    def _init_grid(self, grid_col, grid_row):
        """
        Generates a grid of positions to evaluate the KDE models.
        :param grid_col: Number of elements in the columns axis in the grid to evaluate the KDE models.
        :param grid_row: Number of elements in the rows axis in the grid to evaluate the KDE models.
        :return: 2-D array (2x (grid_x*grid_y)) of positions where to evaluate the KDE Models.
        """
        col_flat = np.linspace(0, 32, grid_col)
        row_flat = np.linspace(0, 32, grid_row)
        col, row = np.meshgrid(col_flat, row_flat)
        self.grid_coords = np.append(col.reshape(-1, 1), row.reshape(-1, 1), axis=1)

    def _window_weights(self, test_obs_set):
        """
        Compute the weights of each temporal window. The weight of each temporal window is the relative length of
        each temporal window.
        :param test_obs_set: ObservationSet or ObservationROISet where the temporal window weights are computed.
        :return: Temporal window weights.
        """
        if test_obs_set.num_windows() == 1:
            return np.asarray([1])
        elif test_obs_set.num_windows() == 3:
            weights = np.empty((3,))
            if test_obs_set.has_obstacle1:
                weights[0] = test_obs_set.ini_obstacle1[0]
                weights[1] = test_obs_set.end_obstacle1[0] - test_obs_set.ini_obstacle1[0]
                weights[2] = test_obs_set.length_vector[0] - test_obs_set.end_obstacle1[0]
            elif test_obs_set.has_obstacle2:
                weights[0] = test_obs_set.ini_obstacle2[0]
                weights[1] = test_obs_set.end_obstacle2[0] - test_obs_set.ini_obstacle2[0]
                weights[2] = test_obs_set.length_vector[0] - test_obs_set.end_obstacle2[0]
            else:
                raise ValueError("Something wrong with the window information in the observation set.")
        elif test_obs_set.num_windows() == 5:
            weights = np.empty((5,))
            weights[0] = test_obs_set.ini_obstacle1[0]
            weights[1] = test_obs_set.end_obstacle1[0] - test_obs_set.ini_obstacle1[0]
            weights[2] = test_obs_set.ini_obstacle2[0] - test_obs_set.end_obstacle1[0]
            weights[3] = test_obs_set.end_obstacle2[0] - test_obs_set.ini_obstacle2[0]
            weights[4] = test_obs_set.length_vector[0] - test_obs_set.end_obstacle2[0]
        else:
            raise ValueError("Number of windows is different from expected.")

        return weights / test_obs_set.length_vector[0]

    def evaluate_observation(self, test_obs_set):
        """
        Returns the anomaly score for the test observation test_obs_set as the weighted Kullback-Leibler for each
        temporal window.

        :param obs_test: ObservationSet or ObservationROISet to classify.
        :return: Anomaly score of test_obs_set.
        """
        test_score = np.empty((test_obs_set.num_windows(),))
        for n_window in range(0, test_obs_set.num_windows()):
            obs_window = test_obs_set.get_window_observation(n_window)
            kde_model = stats.gaussian_kde(obs_window.get_valid_window_array())
            evaluation = kde_model(self.grid_coords.T)
            # Kullback-Leibler for each temporal window.
            test_score[n_window] = stats.entropy(self.train_grid[n_window], evaluation)

        weights = self._window_weights(test_obs_set)
        return np.average(test_score, weights=weights)

def evaluate_model(data_folder, output_name):
    """
    Evaluates the GlobalKDE for the data using a hold-out strategy. All the videos are tested using the first
    4 videos as training data. Then, the first 4 videos are tested using the next 4 videos as training data.

    :param data_folder: Folder name where the data is located.
    :param output_name: Name of the files which contains the result of the GlobalKDE classifier.
    :return:
    """
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)
    obs_set.synchronize_average()

    kde_results = np.empty(obs_set.num_observations())
    kde_results_gaussian = np.empty(obs_set.num_observations())

    # Using the first 4 videos as train data.
    observations_train = obs_set.select_observations([0,1,2,3], inplace=False)
    kde_model = GlobalKDE(observations_train)
    for idx_test,n_test in enumerate(range(4,obs_set.num_observations())):
        print(str(idx_test) + ' / ' + str(obs_set.num_observations()))
        observations_test = obs_set.select_observations(n_test, inplace=False)
        # Generate noisy data.
        observations_gaussian = observations_test.gaussian_noise(std_col=0.141421356, std_row=0.141421356, inplace=False)

        # Evaluate the anomaly score.
        kde_results[n_test] = kde_model.evaluate_observation(observations_test)
        kde_results_gaussian[n_test] = kde_model.evaluate_observation(observations_gaussian)

    # Using the next 4 videos as train data to classify the first 4 videos.
    observations_train = obs_set.select_observations([4,5,6,7], inplace=False)
    kde_model = GlobalKDE(observations_train)
    for idx_test,n_test in enumerate(range(0,4)):
        print(str(idx_test) + str(obs_set.num_observations() - 4) + ' / ' + str(obs_set.num_observations()))
        observations_test = obs_set.select_observations(n_test, inplace=False)
        # Generate noisy data.
        observations_gaussian = observations_test.gaussian_noise(std_col=0.141421356, std_row=0.141421356, inplace=False)

        # Evaluate the anomaly score.
        kde_results[n_test] = kde_model.evaluate_observation(observations_test)
        kde_results_gaussian[n_test] = kde_model.evaluate_observation(observations_gaussian)

    # Save the results to the output_name files.
    with open(output_name + '_normal.csv', 'w') as normal_file, \
            open(output_name + '_gaussian002.csv', 'w') as gaussian_file:
        normal_file.write("Name,AnomalyScore" + '\n')
        gaussian_file.write("Name,AnomalyScore" + '\n')
        for n in range(0, obs_set.num_observations()):
            print("Writing " + obs_set.names_vector[n])
            normal_file.write(obs_set.names_vector[n] + ',' + str(kde_results[n]) + '\n')
            gaussian_file.write(obs_set.names_vector[n] + ',' + str(kde_results_gaussian[n]) + '\n')


if __name__ == '__main__':
    if not os.path.isdir('results/GlobalKDE'):
        os.mkdir('results/GlobalKDE')

    for workpiece_type in range(1, 37):
        data_folder = "data/Type" + str(workpiece_type)

        result_folder = "results/GlobalKDE/Type" + str(workpiece_type)
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

        evaluate_model(data_folder, result_folder + "/GlobalKDE")