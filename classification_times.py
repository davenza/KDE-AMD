import observations_set
from observations_set import SymbolizationType, DivisionOrder
from kde_amd import KDEAMD
from global_kde import GlobalKDE
from dmarkov_machine import DMarkovMachine
from kalman_filter import KalmanFilter
from deep_neural_network_positions import DeepNeuralNetworkPosition
import scipy.stats as stats
import time
import numpy as np

def kdeamd_classification_times(n_cell_col, n_cell_row, min_unique_points, data_folder):
    """
    Measures the time needed to evaluate an observation with the KDE-AMD algorithm.

    There are 3 different time measures for this algorithm:
    - evaluate time: time needed to obtain the loglikelihood of each movement.
    - dist array time: time needed to obtain the distribution of the loglikelihood.
    - kl time: time needed to compute the Kullback-Leibler measure.

    :param n_cell_col: Number of regions in the columns axis.
    :param n_cell_row: Number of regions in the rows axis.
    :param min_unique_points: Minimum number of points for each region of the KDE-AMD.
    :param data_folder: Folder name where the data is located.
    :return: evaluate time, dist array time, kl time (in seconds).
    """
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)
    obs_set.synchronize_average()

    dist_array = np.empty((obs_set.num_observations(),), dtype=object)

    xmin = np.inf
    xmax = -np.inf

    classification_time_evaluate = 0
    classification_time_dist_array = 0
    classification_time_kl = 0
    for n in range(0, obs_set.num_observations()):
        observations_train = obs_set.unselect_observations(n, inplace=False)
        observations_test = obs_set.select_observations(n, inplace=False)

        kde_amd = KDEAMD(observations_train, min_unique_points, n_cell_col=n_cell_col, n_cell_row=n_cell_row)

        start = time.time()

        probabilities = kde_amd.evaluate_observation(observations_test)
        logLikelihood = -np.log(probabilities)

        end = time.time()
        classification_time_evaluate += (end - start)

        dist_array[n] = stats.gaussian_kde(logLikelihood)

        if xmin > logLikelihood.min():
            xmin = logLikelihood.min()
        if xmax < logLikelihood.max():
            xmax = logLikelihood.max()

    # Generates the distribution of the loglikelihood.
    range_loglikelihood = np.linspace(xmin,xmax,2000)
    distribution_array = np.empty((obs_set.num_observations(), 2000))

    # Evaluates the distribution of the loglikelihood for the normal/gaussian noise data.
    for n in range(0, obs_set.num_observations()):
        start = time.time()

        distribution_array[n] = dist_array[n](range_loglikelihood)

        end = time.time()
        classification_time_dist_array += (end - start)

    for n in range(0, obs_set.num_observations()):
        distributions_train = np.delete(distribution_array, n, axis=0)
        # The "correct" distribution of the loglikelihood is considered to be the mean of the distributions.
        mean_train = distributions_train.mean(axis=0)

        start = time.time()

        # Compute the Kullback-Leibler with respect the "correct" distribution of the loglikelihood.
        mean_kl_value = stats.entropy(mean_train, distribution_array[n])

        end = time.time()
        classification_time_kl += (end - start)

    return classification_time_evaluate / obs_set.num_observations(), \
           classification_time_dist_array / obs_set.num_observations(), \
           classification_time_kl / obs_set.num_observations()

def global_kde_classification_times(data_folder):
    """
    Measures the time needed to classify an observation with the Global KDE algorithm. It uses 4 observations to train
    the Global KDE algorithm.

    :param data_folder: Folder name where the data is located.
    :return: Time needed to evaluate the Global KDE algorithm (in seconds).
    """
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)
    obs_set.synchronize_average()

    kde_results = np.empty(obs_set.num_observations())

    # Using the first 4 videos as train data.
    observations_train = obs_set.select_observations([0,1,2,3], inplace=False)
    KDE_model = GlobalKDE(observations_train)
    classification_time = 0
    for idx_test,n_test in enumerate(range(4,obs_set.num_observations())):
        print(str(idx_test) + ' / ' + str(obs_set.num_observations()))
        observations_test = obs_set.select_observations(n_test, inplace=False)

        # Evaluate the anomaly score.
        start = time.time()
        kde_results[n_test] = KDE_model.evaluate_observation(observations_test)
        end = time.time()
        classification_time += end-start
    return classification_time / (obs_set.num_observations() - 4)

def dmarkov_classification_times(n_cell_col, n_cell_row, D, symbolizationType, division_order, data_folder):
    """
    Measures the time needed to classify an observation with the D-Markov algorithm.

    :param n_cell_col: Number of regions in the columns axis.
    :param n_cell_row: Number of regions in the rows axis.
    :param D: Number of previous symbols to take into account (Markov property).
    :param symbolizationType: Type of symbolization. It should be an Enum of type SymbolizationType (observations_set.py)
                (see EqualWidthLimits, EqualFrequencyLimits and EqualFrequencyLimitsNoBounds in observations_set.py).
    :param division_order: Only for EqualFrequencyLimits and EqualFrequencyLimitsNoBounds. Should we do a row-first
        or column-first division? It should be an Enum of type DivisionOrder (observations_set.py)
    :param data_folder: Folder name where the data is located.
    :return: Time needed to evaluate the D-Markov algorithm (in seconds).
    """
    # Load data
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)
    obs_set.synchronize_average()

    score = np.empty((obs_set.num_observations(),))

    classification_time = 0
    for n in range(0, obs_set.num_observations()):
        print(str(n) + " / " + str(obs_set.num_observations()) + " " + obs_set.names_vector[n])
        observations_train = obs_set.unselect_observations(n, inplace=False)

        observations_test = obs_set.select_observations(n, inplace=False)

        # Symbolize the data and return the same method to symbolize the test data.
        train_symbol_set, symbolizer = observations_set.SymbolSet.from_observationset(observations_train, n_cell_col, n_cell_row,
                                                                                      symbolizationType,
                                                                                      division_order)

        start = time.time()
        # Symbolize the test data.
        test_symbol_set = observations_set.SymbolSet(symbolizer.symbolizate(observations_test),
                                                     observations_test.length_vector,
                                                     observations_test.names_vector)
        end = time.time()
        classification_time += end-start
        # Train the D-Markov machine classifier and compute the anomaly score.
        DMarkov = DMarkovMachine(train_symbol_set, D)

        start = time.time()
        score[n] = DMarkov.evaluate_observation(test_symbol_set)
        end = time.time()
        classification_time += end-start
    return classification_time / obs_set.num_observations()

def kalman_filter_classification_times(data_folder):
    """
    Measures the time needed to classify an observation with the Kalman filter algorithm.

    :param data_folder: Folder name where the data is located.
    :return: Time needed to evaluate the Kalman filter algorithm (in seconds).
    """
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)
    obs_set.synchronize_average()

    Fk = np.eye(4)
    Fk[0, 2] = 1
    Fk[1, 3] = 1

    Hk = np.eye(4)

    Qk = np.eye(4)
    Rk = np.eye(4)
    Pk_minus = np.eye(4)
    initial_status = np.asarray([16, 16, 0, 0])

    # Creates the Kalman filter.
    kalmanModel = KalmanFilter(Fk, Hk, Qk, Rk, Pk_minus, initial_status)

    score = np.empty((obs_set.num_observations(),))

    classification_time = 0
    for n in range(obs_set.num_observations()):
        print(str(n) + " / " + str(obs_set.num_observations()) + " " + obs_set.names_vector[n])
        observations_test = obs_set.select_observations(n, inplace=False)

        # Evaluates the anomaly score for the normal/normal noise data.
        start = time.time()
        score[n] = kalmanModel.evaluate_observation(observations_test)
        end = time.time()
        classification_time += end-start
    return classification_time / obs_set.num_observations()

def nn_positions_classification_times(weights, data_folder):
    """
    Measures the time needed to classify an observation with the deep learning network.

    :param weights: Name of weights model to load.
    :param data_folder: Folder name where the data is located.
    :return: Time needed to evaluate the deep learning network (in seconds).
    """
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)
    obs_set.synchronize_average()

    nn = DeepNeuralNetworkPosition.from_trained_model(weights)

    normal_results = np.empty((obs_set.num_observations(),))

    classification_time = 0
    for i in range(obs_set.num_observations()):
        observations_test = obs_set.select_observations(i, inplace=False)
        start = time.time()
        normal_results[i] = nn.evaluate_observation(observations_test)
        end = time.time()
        classification_time += end-start

    return classification_time / obs_set.num_observations()

if __name__ == "__main__":
    evaluate, dist, kl = kdeamd_classification_times(35, 35, 200, 'data/Type1')
    print("Classification time for KDE-AMD 35x35 lambda " + str(200) + ": ")
    print("\t Evaluate: " + str(evaluate))
    print("\t Distribution: " + str(dist))
    print("\t KL: " + str(kl))

    windows_list = [16, 20, 25, 30, 35, 40]
    lambda_list = [5, 10, 15, 20, 30, 40, 50, 200]

    for i in windows_list:
        evaluate, dist, kl = kdeamd_classification_times(i, i, 5, 'data/Type1')
        print("Classification time for KDE-AMD " + str(i) + "x" + str(i) + " lambda 5: ")
        print("\t Evaluate: " + str(evaluate))
        print("\t Distribution: " + str(dist))
        print("\t KL: " + str(kl))

    for i in lambda_list[1:]:
        evaluate, dist, kl = kdeamd_classification_times(35, 35, i, 'data/Type1')
        print("Classification time for KDE-AMD 35x35 lambda " + str(i) + ": ")
        print("\t Evaluate: " + str(evaluate))
        print("\t Distribution: " + str(dist))
        print("\t KL: " + str(kl))

    classification_time = global_kde_classification_times('data/Type1')
    print("Classification time for Global KDE:")
    print("\tEvaluate: " + str(classification_time))

    classification_time = kalman_filter_classification_times('data/Type1')
    print("Classification time for Kalman Filter:")
    print("\tEvaluate: " + str(classification_time))

    classification_time = nn_positions_classification_times('nn_positions_models/Type1/Type1_first', 'data/Type1')
    print("Classification time for neural network:")
    print("\tEvaluate: " + str(classification_time))


    symbolization = [(SymbolizationType.EQUAL_WIDTH, 'EW'),
                     (SymbolizationType.EQUAL_FREQUENCY, 'EF'),
                     (SymbolizationType.EQUAL_FREQUENCY_NO_BOUNDS, 'EFNB')]


    division_order = [(DivisionOrder.ROWS_THEN_COLUMNS, 'RC'),
                      (DivisionOrder.COLUMNS_THEN_ROWS, 'CR')]

    for sym_process in symbolization:
        if sym_process[0] == SymbolizationType.EQUAL_WIDTH:
            print("Executing with 40x40 D = 1, symbolization = " + sym_process[1])
            classification_time = dmarkov_classification_times(40, 40, 1, sym_process[0], None, 'data/Type1')
            print("Classification time for D-Markov 40x40 D = 1, symbolization = " + sym_process[1])
            print("\tEvaluate: " + str(classification_time))
        else:
            for division_process in division_order:
                print("Executing with 40x40 D = 1, symbolization = " + sym_process[1] + " division_order = " + division_process[1])
                classification_time = dmarkov_classification_times(40, 40, 1, sym_process[0], division_process[0], 'data/Type1')
                print("Classification time for D-Markov 40x40 D = 1, symbolization = " + sym_process[1] + " division_order = " + division_process[1])
                print("\tEvaluate: " + str(classification_time))