"""
This file implements a D-Markov machine to detect anomalies.
"""

import numpy as np
import observations_set

import os
import numpy.lib.stride_tricks as stride_tricks


class DMarkovMachine(object):
    """
    D-Markov machine classifier.
    """
    def __init__(self, train_symbol_set, D):
        """
        Trains a D-Markov machine classifier using a symbol_set and the parameter D.
        :param train_symbol_set: A SymbolSet (see observations_set.py) of the training data. It contains the symbol
        asssociated to each observation in the training set.
        :param D: Number of previous symbols to take into account (Markov property).
        """
        self.D = D
        # Computes the trainstion matrix.
        matrix, self.train_statesD = self.stochastic_matrix(train_symbol_set, D)

        # Extract the eigenvectors.
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        index_prob_vector = np.where(np.isclose(eigenvalues, 1))[0]
        if np.iscomplex(eigenvectors[:,index_prob_vector]).any():
            raise ValueError("complex value in prob vector.")

        # The expected probability of each state is in the eigenvectors.
        self.train_prob_vector = eigenvectors[:,index_prob_vector].real

    def stochastic_matrix(self, train_symbol_set, D):
        """
        Return the stochastic (transition) matrix of a given SymbolSet with D-Markov property.
        :param D: Number of previous symbols to take into account (Markov property).
        :param train_symbol_set: A SymbolSet (see observations_set.py) of the training data. It contains the symbol
        asssociated to each observation in the training set.
        :return: Stochastic matrix where the rows sum to 1,
                2-D array containing the unique states of the stochastic matrix. Each row contains a different state.
        """
        # Get the unique states (ant its count) as a concatenation of symbols of D+1 length.
        statesD_extend, countD_extend = self._get_states_order(train_symbol_set, D + 1)
        # Get the unique states of a concatenation of symbols of D length.
        statesD = statesD_extend[:, :D]
        uniq_statesD = np.unique(statesD, axis=0)

        # Checks the destination states are included in uniq_statesD. If they are not included, they cannot be included
        # in the transition matrix.
        idx_dest_non_existent = set()
        keep_removing_states = True
        while keep_removing_states and len(idx_dest_non_existent) != uniq_statesD.shape[0]:
            idx_tmp_delete = []
            keep_removing_states = False
            for n, from_state in enumerate(uniq_statesD):
                from_idx = self._find_row(statesD, from_state)
                dest_states = statesD_extend[from_idx][:, 1:]
                dest_valid, dest_mat_id = self._filter_states(uniq_statesD, dest_states)
                if dest_valid.shape[0] == 0:
                    keep_removing_states = True
                    idx_dest_non_existent.add(n)
                    idx_tmp_delete.append(n)

            uniq_statesD = np.delete(uniq_statesD, idx_tmp_delete,0)

        matrix = np.zeros((uniq_statesD.shape[0], uniq_statesD.shape[0]), dtype=np.float)
        for n, from_state in enumerate(uniq_statesD):
            from_idx = self._find_row(statesD, from_state)
            dest_states = statesD_extend[from_idx, 1:]
            dest_valid, dest_mat_id = self._filter_states(uniq_statesD, dest_states)
            numD_extend = countD_extend[from_idx[dest_valid]]
            matrix[n, dest_mat_id] = numD_extend / numD_extend.sum().astype(np.float)

        return matrix, uniq_statesD

    def _find_row(self, matrix2d, row):
        """
        Returns the indices where row is located in matrix2d.
        :param matrix2d: A 2-D matrix.
        :param row: A 1-D array.
        :return: The indices where row is included in matrix2d. Empty array if the row is not found.
        """
        return np.where((matrix2d == row).all(axis=1))[0]

    def _filter_states(self, indices_matrix, rows):
        """
        Filter the rows to find those included in indices_matrix.
        :param indices_matrix: A 2-D matrix.
        :param rows: Rows to be filtered.
        :return: The indices for the rows matrix that contains the included rows in indices_matrix,
                The indices for the indices_matrix that contains the included rows.
        """
        if rows.ndim == 1:
            idx_row = self._find_row(indices_matrix, rows)
            if idx_row.shape[0] != 0:
                return rows, idx_row
            return np.asarray([]), np.asarray([])

        elif rows.ndim == 2:
            idx_indices = []
            idx_rows = []
            for n, row in enumerate(rows):
                idx_indices_row = self._find_row(indices_matrix, row)
                if idx_indices_row.shape[0] != 0:
                    idx_rows.append(n)
                    idx_indices.append(idx_indices_row[0])

            return np.asarray(idx_rows), np.asarray(idx_indices)

    def _get_states_order(self, train_symbol_set, D):
        """
        Returns an array view of train_symbol_set where all overlapping fragments of D states are shown. This method
        removes all fragments where there is a bound involved (contains a -1 symbol).
        :param train_symbol_set: A SymbolSet
        :param D: Length of each fragment.
        :return: 2D array of [N x D] where N is the number of overlapping, fragments of size D.
        """
        symbols_shape = train_symbol_set.symbols.shape
        symb_stride = stride_tricks.as_strided(train_symbol_set.symbols, shape=(symbols_shape[0] * symbols_shape[1] - D + 1, D),
                         strides=(train_symbol_set.symbols.dtype.itemsize, train_symbol_set.symbols.dtype.itemsize))

        # Delete states idx with -1 symbols
        valid_states_idx = np.where((symb_stride != -1).all(axis=1))[0]

        # Get unique states
        uniq, count = np.unique(symb_stride[valid_states_idx], return_counts=True, axis=0)
        return uniq, count

    def evaluate_observation(self, test_symbol_set):
        """
        Returns the anomaly score for the test_symbol_set.
        :param test_symbol_set: A SymbolSet of test data.
        :return: Anomaly score for test_symbol_set
        """
        # Get the stochastic matrix and expected state probability for the test data.
        matrix, test_statesD = self.stochastic_matrix(test_symbol_set, self.D)

        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        index_prob_vector = np.where(np.isclose(eigenvalues, 1))[0]
        if np.iscomplex(eigenvectors[:,index_prob_vector]).any():
            raise ValueError("complex value in prob vector.")
        test_prob_vector = eigenvectors[:,index_prob_vector].real

        idx_test, idx_train = self._filter_states(self.train_statesD, test_statesD)
        # Common states in the train / test data.
        common_train = self.train_prob_vector[idx_train]
        common_test = test_prob_vector[idx_test]

        # Non-common states in the train / test data.
        non_common_train = np.delete(self.train_prob_vector, idx_train, axis=0)
        non_common_test = np.delete(test_prob_vector, idx_test, axis=0)

        # Non-common states are treated as if its expected probability is zero.
        anomaly_score = np.linalg.norm(common_train - common_test) + \
                        np.linalg.norm(non_common_train - np.zeros(non_common_train.shape)) + \
                        np.linalg.norm(non_common_test - np.zeros(non_common_test.shape))

        return anomaly_score

def evaluate_model(n_cell_col, n_cell_row, D, symbolization_type, division_order, data_folder, output_name):
    """
    Applies leaving-one-out training a D-Markov machine with n_cell_col divisions on the columns axis and
    n_cell_row divisions on the rows axis (See the DMarkovClassifier class, also for the D parameter).

    The type of symbolization and the division order for the selected order can also be selected.

    folder is the folder name of the data. output_name is the name of the file where the results will be saved.
    :param n_cell_col: Number of regions in the columns axis.
    :param n_cell_row: Number of regions in the rows axis.
    :param D: Number of previous symbols to take into account (Markov property).
    :param symbolization_type: Type of symbolization. It should be an Enum of type SymbolizationType (observations_set.py)
                (see EqualWidthLimits, EqualFrequencyLimits and EqualFrequencyLimitsNoBounds in observations_set.py).
    :param division_order: Only for EqualFrequencyLimits and EqualFrequencyLimitsNoBounds. Should we do a row-first
        or column-first division? It should be an Enum of type DivisionOrder (observations_set.py)
    :param data_folder: Folder name where the data is located.
    :param output_name: Name of the files which contains the result of the D-Markov machine using leaving-one-out.
    :return:
    """
    # Load data
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)
    obs_set.synchronize_average()

    score = np.empty((obs_set.num_observations(),))
    score_gaussian = np.empty((obs_set.num_observations(),))

    for n in range(0, obs_set.num_observations()):
        print(str(n) + " / " + str(obs_set.num_observations()) + " " + obs_set.names_vector[n])
        observations_train = obs_set.unselect_observations(n, inplace=False)

        observations_test = obs_set.select_observations(n, inplace=False)
        # Generate Gaussian noise.
        observationsGaussian = observations_test.gaussian_noise(std_col=0.141421356, std_row=0.141421356, inplace=False)

        # Symbolize the data and return the same method to symbolize the test data.
        train_symbol_set, symbolizer = observations_set.SymbolSet.from_observationset(observations_train, n_cell_col, n_cell_row,
                                                                                      symbolization_type,
                                                                                      division_order)

        # Symbolize the test data.
        test_symbol_set = observations_set.SymbolSet(symbolizer.symbolizate(observations_test),
                                                     observations_test.length_vector,
                                                     observations_test.names_vector)

        gaussian_symbol_set = observations_set.SymbolSet(symbolizer.symbolizate(observationsGaussian),
                                                          observationsGaussian.length_vector,
                                                          observationsGaussian.names_vector)
        # Train the D-Markov machine classifier and compute the anomaly score.
        DMarkov = DMarkovMachine(train_symbol_set, D)

        score[n] = DMarkov.evaluate_observation(test_symbol_set)
        score_gaussian[n] = DMarkov.evaluate_observation(gaussian_symbol_set)

    # Save the anomaly score in the ouput_name files.
    with open(output_name + '_normal.csv', 'w') as normal_file, \
            open(output_name + '_gaussian002.csv', 'w') as gaussian_file:
        normal_file.write("Name,AnomalyScore" + '\n')
        gaussian_file.write("Name,AnomalyScore" + '\n')
        for n in range(0, obs_set.num_observations()):
            print("Writing " + obs_set.names_vector[n])
            normal_file.write(obs_set.names_vector[n] + ',' + str(score[n]) + '\n')
            gaussian_file.write(obs_set.names_vector[n] + ',' + str(score_gaussian[n]) + '\n')


if __name__ == '__main__':
    windows = 35
    D = 1

    symbolization = [(observations_set.SymbolizationType.EQUAL_WIDTH, 'EW'),
                     (observations_set.SymbolizationType.EQUAL_FREQUENCY, 'EF'),
                     (observations_set.SymbolizationType.EQUAL_FREQUENCY_NO_BOUNDS, 'EFNB')]

    division_order = [(observations_set.DivisionOrder.ROWS_THEN_COLUMNS, 'RC'),
                      (observations_set.DivisionOrder.COLUMNS_THEN_ROWS, 'CR')]

    if not os.path.isdir('results/DMarkovMachine'):
        os.mkdir('results/DMarkovMachine')

    # Evaluate the algorithm for each batch data, symbolization type and division_order possible.
    for workpiece_type in range(1, 37):
        data_folder = "data/Type" + str(workpiece_type)

        result_folder = "results/DMarkovMachine/Type" + str(workpiece_type)
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

        for sym_process in symbolization:
            if sym_process[0] == observations_set.SymbolizationType.EQUAL_WIDTH:
                print("Executing with " + str(windows) + "x" + str(windows) + ", D = " + str(D) +
                      ", symbolization = " + sym_process[1])
                evaluate_model(windows, windows, D, sym_process[0], observations_set.DivisionOrder.ROWS_THEN_COLUMNS,
                               data_folder, result_folder + "/DMarkovMachine_" + str(windows) + "_" + str(windows) +
                              "_" + str(D) + "_" + sym_process[1])
            else:
                for division_process in division_order:
                    print("Executing with " + str(windows) + "x" + str(windows) + ", D = " + str(D) +
                          ", symbolization = " + sym_process[1] + " and division_order = " + division_process[1])
                    evaluate_model(windows, windows, D, sym_process[0], division_process[0], data_folder,
                                   result_folder + "/DMarkovMachine_" + str(windows) + "_" + str(windows) +
                                  "_" + str(D) + "_" + sym_process[1] + "_" + division_process[1])