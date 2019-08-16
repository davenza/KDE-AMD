"""
This file implements a deep neural network that tries to predict the next position of the laser spot from the current
position of the laser spot.
"""
import numpy as np
import observations_set

from keras import Sequential
from keras.layers import Dense
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import random
import os

import glob

class DeepNeuralNetworkPosition(object):
    """
    Deep neural network classifier.
    """
    def __init__(self, nn_model):
        """
        Initializes the deep neural network from a Keras Model.
        :param nn_model:
        """
        self.nn = nn_model

    @classmethod
    def from_data(cls, obs_set, weights_filename, batch_size=256, epochs=100):
        """
        Train the deep neural network from the data of an ObservationROISet.

        :param obs_set: Training ObservationROISet.
        :param weights_filename: filename prefix for saving the structure/weights of the trained deep neural network.
        :param batch_size: batch size for training.
        :param epochs: number of epochs to train.
        :return: A trained DeepNeuralNetworkPosition.
        """
        origin_positions, dest_positions = DeepNeuralNetworkPosition._generate_movements(obs_set)

        origin_positions /= 32
        dest_positions /= 32

        nn = DeepNeuralNetworkPosition._train_model(origin_positions, dest_positions, weights_filename, batch_size=batch_size, epochs=epochs)
        return DeepNeuralNetworkPosition(nn)

    @classmethod
    def from_trained_model(cls, name):
        """
        Loads a pretrained model given the filename prefix for the structure/weights. When there is more than one weights
        file for a given structure, the last epoch weights will be selected because only the epochs that improve the
        evaluation loss are saved.
        :param name: filename prefix of the model.
        :return: A trained DeepNeuralNetworkPosition
        """
        with open(name + '-structure.json', 'r') as structure:
            model = model_from_json(structure.read())

        weight_files = glob.glob(name + "*.hdf5")

        greater_epoch_index = 0
        last_epoch = 0
        for i, f in enumerate(weight_files):
            start_epoch_string = len(name)+1
            epoch = int(f[start_epoch_string:(start_epoch_string+3)])
            if epoch > last_epoch:
                last_epoch = epoch
                greater_epoch_index = i

        best_weights_file = weight_files[greater_epoch_index]
        model.load_weights(best_weights_file)
        return DeepNeuralNetworkPosition(model)

    @classmethod
    def _generate_movements(cls, obs_set):
        """
        Generate the movements returning the origin and destination points of each movement.
        :param obs_set: ObservationROISet.
        :return: origin positions, destination positions of each movement
        """
        num_movements = 0

        for i in range(obs_set.num_observations()):
            num_movements += obs_set.length_vector[i] - 1

        origin_positions = np.empty((num_movements, 2))
        dest_positions = np.empty((num_movements, 2))

        current_pos = 0
        for i in range(obs_set.num_observations()):
            length = obs_set.length_vector[i]
            origin_positions[current_pos:(current_pos + length - 1), :] = obs_set.obs[i, :, :(length - 1)].T
            dest_positions[current_pos:(current_pos + length - 1), :] = obs_set.obs[i, :, 1:length].T
            current_pos += length - 1

        return origin_positions, dest_positions

    @classmethod
    def _generate_model(cls, weights_filename):
        """
        Generates the structure of the deep neural network.

        :param weights_filename: filename prefix to save the structure.
        :return: model structure.
        """
        model = Sequential()

        model.add(Dense(8, activation='relu', input_shape=(2,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))

        model.summary()

        with open(weights_filename + '-structure.json', 'w') as model_json:
            model_json.write(model.to_json())

        return model

    @classmethod
    def _train_model(cls, origin_positions, dest_positions, weights_filename, batch_size=256, epochs=100):
        """
        Train a deep neural network given the origin and destination points of a set of movements. A set of movements
        (20%) is selected randomly as validation data.

        The weights of the model are only saved for those epochs that improve the validation loss (mean squared error).

        :param origin_positions: Origin positions of the movements set.
        :param dest_positions: Destination positions of the movements set.
        :param weights_filename: filename prefix to save the structure/weights.
        :param batch_size: batch size to train the deep neural network.
        :param epochs: number of epochs to train the deep neural network.
        :return:
        """
        model = DeepNeuralNetworkPosition._generate_model(weights_filename)

        nn = Model(inputs=model.input, outputs=model.output)

        nn.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

        train_origin, valid_origin, train_dest, valid_dest = train_test_split(origin_positions,
                                                                                dest_positions,
                                                                                test_size=0.2,
                                                                                random_state=13)

        logger = ModelCheckpoint(weights_filename + "-{epoch:03d}-{val_loss:.6f}.hdf5", monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='min', period=1)

        nn.fit(train_origin, train_dest, batch_size=batch_size, epochs=epochs, verbose=1,
               validation_data=(valid_origin, valid_dest), callbacks=[logger])

        return nn

    def evaluate_observation(self, obs_test):
        """
        Returns the anomaly score for a given test ObservationROISet.
        :param obs_test: ObservationROISet to test.
        :return: anomaly score.
        """
        origin_test, dest_test = self._generate_movements(obs_test)
        origin_test /= 32
        dest_test /= 32
        predicted = self.nn.predict(origin_test)
        diff = dest_test - predicted
        hypot_distance = np.hypot(diff[:,0], diff[:,1])
        return hypot_distance.sum()

def evaluate_model(data_folder, weights_filename, output_name):
    """
    Applies a 2-fold cross validation to evaluate the performance of the deep neural network.

    :param data_folder: Folder name where the data is located.
    :param weights_filename: filename prefix to save the structure/weights.
    :param output_name: Name of the files which contains the result of the deep nueral network using leaving-one-out.
    :return:
    """
    obs_set = observations_set.ObservationROISet.fromfolder(data_folder)
    obs_set.synchronize_average()

    possible_train_indices = range(0, obs_set.num_observations())
    # Don't train with the known anomaly.
    if "1673" in obs_set.names_vector:
        anomaly_index = np.where(obs_set.names_vector == "1673")[0][0]
        possible_train_indices = list(possible_train_indices)
        del possible_train_indices[anomaly_index]

    num_test = int(0.5*obs_set.num_observations())
    random.seed(0)

    train_idx_first = np.asarray(random.sample(possible_train_indices, num_test))
    train_idx_second = list(set(possible_train_indices) - set(train_idx_first))

    # Generate the train/test sets for the first validation
    train_obs_first = obs_set.select_observations(train_idx_first, inplace=False)
    test_obs_first = obs_set.unselect_observations(train_idx_first, inplace=False)

    nn = DeepNeuralNetworkPosition.from_data(train_obs_first, weights_filename + "_first")

    normal_results = np.empty((obs_set.num_observations(),))
    gaussian_results = np.empty((obs_set.num_observations(),))

    for i in range(test_obs_first.num_observations()):
        test_observation = test_obs_first.select_observations(i, inplace=False)
        # Generate noise in the data
        gaussian_observation = test_observation.gaussian_noise(std_col=0.141421356, std_row=0.141421356, inplace=False)

        name = test_obs_first.names_vector[i]
        obs_index = np.where(obs_set.names_vector == name)[0][0]

        normal_results[obs_index] = nn.evaluate_observation(test_observation)
        gaussian_results[obs_index] = nn.evaluate_observation(gaussian_observation)

    # Generate the train/test sets for the first validation
    train_obs_second = obs_set.select_observations(train_idx_second, inplace=False)
    test_obs_second = obs_set.unselect_observations(train_idx_second, inplace=False)

    nn = DeepNeuralNetworkPosition.from_data(train_obs_second, weights_filename + "_second")

    for i in range(test_obs_second.num_observations()):
        test_observation = test_obs_second.select_observations(i, inplace=False)
        # Generate noise in the data
        gaussian_observation = test_observation.gaussian_noise(std_col=0.141421356, std_row=0.141421356, inplace=False)

        name = test_obs_second.names_vector[i]
        obs_index = np.where(obs_set.names_vector == name)[0][0]

        normal_results[obs_index] = nn.evaluate_observation(test_observation)
        gaussian_results[obs_index] = nn.evaluate_observation(gaussian_observation)

    with open(output_name + '_normal.csv', 'w') as normal_file, open(output_name + '_gaussian002.csv', 'w') as gaussian_file:
        normal_file.write("Name,AnomalyScore" + '\n')
        gaussian_file.write("Name,AnomalyScore" + '\n')
        for n in range(0, obs_set.num_observations()):
            # Writes the results.
            normal_file.write(obs_set.names_vector[n] + "," + str(normal_results[n]) + '\n')
            gaussian_file.write(obs_set.names_vector[n] + "," + str(gaussian_results[n]) + '\n')

if __name__ == '__main__':
    if not os.path.isdir('results/DeepNeuralNetworkPosition'):
        os.mkdir('results/DeepNeuralNetworkPosition')

    for t in range(1,37):
        data_folder = 'data/Type' + str(t)
        weights_folder = "nn_positions_models/Type" + str(t)
        result_folder = "results/DeepNeuralNetworkPosition/Type" + str(t)

        if not os.path.isdir(weights_folder):
            os.mkdir(weights_folder)

        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

        evaluate_model(data_folder, weights_folder + "/Type" + str(t), result_folder + "/DeepNeuralNetworkPosition")