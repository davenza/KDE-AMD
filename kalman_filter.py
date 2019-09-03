"""
Implements a Kalman filter to detect anomalies.
"""
import numpy as np
import observations_set
import os

class KalmanFilter(object):
    """
    Kalman filter classifier.
    """
    def __init__(self, Fk, Hk, Qk, Rk, Pk_minus, initial_status):
        """
        Returns a Kalman filter with the initialized parameters.

        For states x_k and observations z_k

        :param Fk: Transition state matrix: x_k = Fk * x_{k-1} + v_k where v_k is noise.
        :param Hk: Transformation matrix from the states to the observations. z_k = Hk * x_k + n_k where n_k is noise.
        :param Qk: Covariance matrix of the v_k noise.
        :param Rk: Covariance matrix of the n_k noise.
        :param Pk_minus: Uncertainty matrix for k-1. Pk = Fk * PkMinus * Fk^T + Qk
        :param initial_status: Initial state.
        """
        self.Fk = Fk
        self.Hk = Hk
        self.Qk = Qk
        self.Rk = Rk
        self.Pk_minus = Pk_minus
        self.initial_status = initial_status

    def evaluate_observation(self, test_obs_set):
        """
        Evaluates a test ObservationSet. This method assumes that the ObservationSet contains a single video
        observation.
        :param test_obs_set: ObservationSet or ObservationROISet to classify.
        :return: Anomaly score for the test_obs_set.
        """

        Pk_minus = self.Pk_minus
        current_status = self.initial_status

        speed = self._get_speed(test_obs_set)
        observations = np.row_stack((test_obs_set.obs[0], speed[0]))
        difference = 0

        # Iterate over each temporal position updating the Kalman filter parameters.
        for t in range(0, test_obs_set.length_vector[0]):
            prediction_status = self.Fk.dot(current_status)
            distance = prediction_status[0:2] - observations[0:2, t]

            # Compute the anomaly score using the distance between the prediction and the actual values for the states.
            difference += (np.exp(distance.dot(np.linalg.inv(Pk_minus[0:2, 0:2])).dot(distance) * (-0.5))) / (
                    np.sqrt(np.linalg.det(Pk_minus[0:2, 0:2])) * 2 * np.pi)

            Pk = self.Fk.dot(Pk_minus).dot(self.Fk.T) + self.Qk
            kalman_gain = Pk.dot(self.Hk.T).dot(np.linalg.inv(self.Hk.dot(Pk).dot(self.Hk.T) + self.Rk))

            current_status = prediction_status + kalman_gain.dot(observations[:, t] - self.Hk.dot(prediction_status))

            Pk_minus = Pk - kalman_gain.dot(self.Hk).dot(Pk)

        return difference

    def _get_speed(self, observations):
        """
        Returns the speed of the observations for a given the ObservationSet. The return type is an array with a shape
        equal to the ObservationSet. In t = 0, the speed value will be setted to 0.
        :param observations: An ObservationSet
        :return: A 2-D array (2x(N-1)) for the speed of the observations in the columns axis [0,:] and rows axis [1,:].
        """
        speed = np.zeros(observations.obs.shape)

        for n in range(observations.num_observations()):
            speed[n,:,1:observations.length_vector[n]] = observations[n,:,1:observations.length_vector[n]] \
                                                           - observations[n,:,:(observations.length_vector[n]-1)]

        return speed

def evaluate_model(Fk, Hk, Qk, Rk, Pk_minus, initial_status, folder, output_name):
    """
    Evaluates the performance of the Kalman Filter computing the anomaly score for each observation.
    :param Fk: See KalmanFilter.
    :param Hk: See KalmanFilter.
    :param Qk: See KalmanFilter.
    :param Rk: See KalmanFilter.
    :param Pk_minus: See KalmanFilter.
    :param initial_status: See KalmanFilter.
    :param data_folder: Folder name where the data is located.
    :param output_name: Name of the files which contains the results of the KalmanFilter.
    :return:
    """
    obs_set = observations_set.ObservationROISet.fromfolder(folder)
    obs_set.synchronize_average()

    # Creates the Kalman filter.
    kalman_model = KalmanFilter(Fk, Hk, Qk, Rk, Pk_minus, initial_status)

    score = np.empty((obs_set.num_observations(),))
    score_gaussian = np.empty((obs_set.num_observations(),))

    for n in range(obs_set.num_observations()):
        print(str(n) + " / " + str(obs_set.num_observations()) + " " + obs_set.names_vector[n])
        observations_test = obs_set.select_observations(n, inplace=False)
        # Generate noisy data.
        observations_gaussian = observations_test.gaussian_noise(std_col=0.141421356, std_row=0.141421356, inplace=False)

        # Evaluates the anomaly score for the normal/normal noise data.
        score[n] = kalman_model.evaluate_observation(observations_test)
        score_gaussian[n] = kalman_model.evaluate_observation(observations_gaussian)

    # Write the results in the output_name files.
    with open(output_name + '_normal.csv', 'w') as normal_file,\
            open(output_name + '_gaussian002.csv', 'w') as gaussian_file:
        normal_file.write("Name,AnomalyScore" + '\n')
        gaussian_file.write("Name,AnomalyScore" + '\n')
        for n in range(0, obs_set.num_observations()):
            print("Writing " + obs_set.names_vector[n])
            normal_file.write(obs_set.names_vector[n] + ',' + str(score[n]) + '\n')
            gaussian_file.write(obs_set.names_vector[n] + ',' + str(score_gaussian[n]) + '\n')

if __name__ == '__main__':
    Fk = np.eye(4)
    Fk[0, 2] = 1
    Fk[1, 3] = 1

    Hk = np.eye(4)

    Qk = np.eye(4)
    Rk = np.eye(4)
    Pk_minus = np.eye(4)
    initial_status = np.asarray([16, 16, 0, 0])

    if not os.path.isdir('results/KalmanFilter'):
        os.mkdir('results/KalmanFilter')

    for workpiece_type in range(1, 37):
        data_folder = "data/Type" + str(workpiece_type)

        result_folder = "results/KalmanFilter/Type" + str(workpiece_type)
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

        evaluate_model(Fk, Hk, Qk, Rk, Pk_minus, initial_status, data_folder, result_folder + "/KalmanFilter")