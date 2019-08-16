import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

from observations_set import SymbolizationType, DivisionOrder
import numpy as np
import tikzplotlib

def plot_roc_curve(table, style=None, label=None):
    """
    Plots a ROC curve given a `table` pandas dataframe with a column called "AnomalyScore". The "AnomalyScore"
    should have larger values for anomalous videos.
    :param table: Pandas dataframe with "AnomalyScore" column.
    :param style: Style for the ROC curve line.
    :param label: Label for the legend of the ROC curve.
    :return: Plots a ROC curve. plt.show() should be called after this function to show the plot.
    """
    if label is None:
        label = ""

    score_values = table['AnomalyScore'].as_matrix()
    true_label = table['Anomalous'].as_matrix()

    fpr, tpr, thresholds = metrics.roc_curve(true_label, score_values)

    if style is None:
        plt.plot(fpr, tpr, label=label + ' (AUC = %0.2f)' % metrics.auc(fpr,tpr))
    else:
        plt.plot(fpr, tpr, style, label=label + ' (AUC = %0.2f)' % metrics.auc(fpr,tpr))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'\LARGE{False Positive Rate}')
    plt.ylabel(r'\LARGE{True Positive Rate}')
    plt.legend(loc="lower right")

def plot_roc_curves(tables, styles=None, labels=None):
    """
    Plots multiple ROC curves given a list of pandas dataframe with a column called "AnomalyScore". The "AnomalyScore"
    should have larger values for anomalous videos.
    :param table: List of pandas dataframes with "AnomalyScore" column.
    :param style: List of styles for the ROC curve lines.
    :param label: List of labels for the legend of the ROC curves.
    :return: Plots multiple ROC curves. plt.show() should be called after this function to show the plot.
    """
    if labels is None:
        labels = []
        for i in range(1, len(tables)+1):
            labels.append("Algorithm " + str(i))

    if styles is None:
        styles = [None]*len(tables)

    for (table, style, label) in zip(tables, styles, labels):
        score_values = table['AnomalyScore'].as_matrix()
        true_label = table['Anomalous'].as_matrix()

        fpr, tpr, thresholds = metrics.roc_curve(true_label, score_values)

        if style is None:
            plt.plot(fpr, tpr, label=label + ' (AUC = %0.5f)' % metrics.auc(fpr, tpr))
        else:
            plt.plot(fpr, tpr, style, label=label + ' (AUC = %0.5f)' % metrics.auc(fpr, tpr))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'\LARGE{False Positive Rate}')
    plt.ylabel(r'\LARGE{True Positive Rate}')
    plt.legend(loc="lower right")

def auc_scores(tables):
    """
    Computes the AUC score for a list of tables with an "AnomalyScore" column.
    :param tables: List of pandas dataframes with "AnomalyScore" column.
    :return: numpy array with the AUC value of each table.
    """
    aucs = []
    for table in tables:
        score_values = table['AnomalyScore'].as_matrix()
        true_label = table['Anomalous'].as_matrix()

        auc = metrics.roc_auc_score(true_label, score_values)
        aucs.append(auc)

    return np.asarray(aucs)

def read_kdeamd(rows, columns, min_unique_points, suffix=["normal", "gaussian002"]):
    """
    Reads the result files for the KDE-AMD algorithm. The function requires a configuration for the parameters of
    the KDE-AMD. The suffix parameter indicates if the non-modified files should be loaded ("normal"), the noisy
    files should be loaded ("gaussian002") or both.

    :param rows: Rows of the KDE-AMD algorithm.
    :param columns: Columns of the KDE-AMD algorithm.
    :param min_unique_points: Minimum number of positions for region.
    :param suffix: Load non-modified ("normal"), noisy ("gaussian002") or both file results.
    :return: A pandas dataframe with the information in the result files. Also, an "Anomalous" column is created, which
    is False for the "normal" result files and True for the "gaussian002" files.
    """
    if isinstance(suffix, str):
        suffix = [suffix]

    basename = 'results/KDEAMD/Type{:d}/KDEAMD_{:d}_{:d}_{:d}_{}.csv'

    kdeamd_df = pd.DataFrame()
    for type_idx in range(1,37):
        for s in suffix:
            normal_name = basename.format(type_idx, rows, columns, min_unique_points, s)
            file_df = pd.read_csv(normal_name, dtype={'Name': 'object', 'AnomalyScore': 'float64'})

            if s == "normal":
                file_df["Anomalous"] = False
            elif s == "gaussian002":
                file_df["Anomalous"] = True

            kdeamd_df = kdeamd_df.append(file_df)

    return kdeamd_df

def read_dmarkov(columns, rows, D, symbolization_type, division_order, suffix=["normal", "gaussian002"]):
    """
    Reads the result files for the D-Markov algorithm. The function requires a configuration for the parameters of
    the D-Markov. The suffix parameter indicates if the non-modified files should be loaded ("normal"), the noisy
    files should be loaded ("gaussian002") or both.

    :param columns: Number of columns in the division.
    :param rows:  Number of rows in the division.
    :param D: Number of previous symbols to take into account (Markov property).
    :param symbolization_type: Type of symbolization. It should be an Enum of type SymbolizationType (observations_set.py)
                (see EqualWidthLimits, EqualFrequencyLimits and EqualFrequencyLimitsNoBounds in observations_set.py).
    :param division_order: Only for EqualFrequencyLimits and EqualFrequencyLimitsNoBounds. Should we do a row-first
        or column-first division? It should be an Enum of type DivisionOrder (observations_set.py)
    :param suffix: Load non-modified ("normal"), noisy ("gaussian002") or both file results.
    :return: A pandas dataframe with the information in the result files. Also, an "Anomalous" column is created, which
    is False for the "normal" result files and True for the "gaussian002" files.
    """
    if isinstance(suffix, str):
        suffix = [suffix]

    basename = 'results/DMarkovMachine/Type{:d}/DMarkovMachine_{:d}_{:d}_{:d}_{}_{}.csv'

    if symbolization_type == SymbolizationType.EQUAL_WIDTH:
        symb_str = "EW"
    else:
        if symbolization_type == SymbolizationType.EQUAL_FREQUENCY:
            symb_str = "EF"
        elif symbolization_type == SymbolizationType.EQUAL_FREQUENCY_NO_BOUNDS:
            symb_str = "EFNB"

        if division_order == DivisionOrder.ROWS_THEN_COLUMNS:
            symb_str += "_RC"
        elif division_order == DivisionOrder.COLUMNS_THEN_ROWS:
            symb_str += "_CR"

    dmarkov_df = pd.DataFrame()
    for type_idx in range(1,37):
        for s in suffix:
            normal_name = basename.format(type_idx, rows, columns, D, symb_str, s)
            file_df = pd.read_csv(normal_name, dtype={'Name': 'object', 'AnomalyScore': 'float64'})

            if s == "normal":
                file_df["Anomalous"] = False
            elif s == "gaussian002":
                file_df["Anomalous"] = True

            dmarkov_df = dmarkov_df.append(file_df)

    return dmarkov_df

def read_kde(suffix=["normal", "gaussian002"]):
    """
    Reads the result files for the Global KDE algorithm. The suffix parameter indicates if the non-modified files
    should be loaded ("normal"), the noisy files should be loaded ("gaussian002") or both.

    :param suffix: Load non-modified ("normal"), noisy ("gaussian002") or both file results.
    :return: A pandas dataframe with the information in the result files. Also, an "Anomalous" column is created, which
    is False for the "normal" result files and True for the "gaussian002" files.
    """
    if isinstance(suffix, str):
        suffix = [suffix]

    basename = 'results/GlobalKDE/Type{:d}/GlobalKDE_{}.csv'

    kde_df = pd.DataFrame()
    for type_idx in range(1,37):
        for s in suffix:
            normal_name = basename.format(type_idx, s)
            file_df = pd.read_csv(normal_name, dtype={'Name': 'object', 'AnomalyScore': 'float64'})

            if s == "normal":
                file_df["Anomalous"] = False
            elif s == "gaussian002":
                file_df["Anomalous"] = True

            kde_df = kde_df.append(file_df)

    return kde_df

def read_kalman(suffix=["normal", "gaussian002"]):
    """
    Reads the result files for the Kalman filter algorithm. The suffix parameter indicates if the non-modified files
    should be loaded ("normal"), the noisy files should be loaded ("gaussian002") or both.

    :param suffix: Load non-modified ("normal"), noisy ("gaussian002") or both file results.
    :return: A pandas dataframe with the information in the result files. Also, an "Anomalous" column is created, which
    is False for the "normal" result files and True for the "gaussian002" files.
    """
    if isinstance(suffix, str):
        suffix = [suffix]

    basename = 'results/KalmanFilter/Type{:d}/KalmanFilter_{}.csv'

    kalman_df = pd.DataFrame()
    for type_idx in range(1,37):
        for s in suffix:
            normal_name = basename.format(type_idx, s)
            file_df = pd.read_csv(normal_name, dtype={'Name': 'object', 'AnomalyScore': 'float64'})

            if s == "normal":
                file_df["Anomalous"] = False
            elif s == "gaussian002":
                file_df["Anomalous"] = True

            kalman_df = kalman_df.append(file_df)

    kalman_df['AnomalyScore'] = 1.0 / kalman_df['AnomalyScore']
    return kalman_df

def read_nn_positions(suffix=["normal", "gaussian002"]):
    """
    Reads the result files for the deep neural network algorithm. The suffix parameter indicates if the non-modified files
    should be loaded ("normal"), the noisy files should be loaded ("gaussian002") or both.

    :param suffix: Load non-modified ("normal"), noisy ("gaussian002") or both file results.
    :return: A pandas dataframe with the information in the result files. Also, an "Anomalous" column is created, which
    is False for the "normal" result files and True for the "gaussian002" files.
    """
    if isinstance(suffix, str):
        suffix = [suffix]

    basename = 'results/DeepNeuralNetworkPosition/Type{:d}/DeepNeuralNetworkPosition_{}.csv'

    nn_position_df = pd.DataFrame()
    for type_idx in range(1,37):
        for s in suffix:
            normal_name = basename.format(type_idx, s)
            file_df = pd.read_csv(normal_name, dtype={'Name': 'object', 'AnomalyScore': 'float64'})

            if s == "normal":
                file_df["Anomalous"] = False
            elif s == "gaussian002":
                file_df["Anomalous"] = True

            nn_position_df = nn_position_df.append(file_df)

    return nn_position_df

def plot_kdeamd_roc():
    """
    Plot all the ROC curves for the KDE-AMD algorithm.
    :return:
    """
    lambda_list = [5, 10, 15, 20, 30, 40, 50, 200]
    windows_list = [16, 20, 25, 30, 35, 40]

    table_list = []
    labels_list = []
    for i in lambda_list[1:]:
        kdeamd_df = read_kdeamd(35, 35, i)
        kdeamd_df = kdeamd_df[kdeamd_df.Name != '1673']
        table_list.append(kdeamd_df)
        labels_list.append('KDE-AMD $35\\times 35$ $\\lambda = {}$'.format(i))

    for i in windows_list:
        kdeamd_df = read_kdeamd(i, i, 5)
        kdeamd_df = kdeamd_df[kdeamd_df.Name != '1673']
        table_list.append(kdeamd_df)
        labels_list.append('KDE-AMD ${}\\times {}$ $\\lambda = 5$'.format(i, i))

    plot_roc_curves(table_list, labels=labels_list)
    plt.show()

def plot_dmarkov_roc():
    """
    Plot all the ROC curves for the D-Markov algorithm.
    :return:
    """
    windows_list = [16, 20, 25, 30, 35, 40]
    D = 1

    symbolization = [(SymbolizationType.EQUAL_WIDTH, 'EW'),
                     (SymbolizationType.EQUAL_FREQUENCY, 'EF'),
                     (SymbolizationType.EQUAL_FREQUENCY_NO_BOUNDS, 'EFNB')]


    division_order = [(DivisionOrder.ROWS_THEN_COLUMNS, 'RC'),
                      (DivisionOrder.COLUMNS_THEN_ROWS, 'CR')]

    dmarkov_tables = []
    labels_list = []
    for w in windows_list:
        for s in symbolization:
            if s[0] == SymbolizationType.EQUAL_WIDTH:
                df = read_dmarkov(w, w, D, s[0], None)
                dmarkov_tables.append(df)
                labels_list.append("D-Markov {}x{} {}".format(w, w, s[1]))
            else:
                for d in division_order:
                    df = read_dmarkov(w, w, D, s[0], d[0])
                    dmarkov_tables.append(df)
                    labels_list.append("D-Markov {}x{} {} {}".format(w, w, s[1], d[1]))

    plot_roc_curves(dmarkov_tables, labels=labels_list)
    plt.show()

def plot_kde_roc():
    """
    Plot the ROC curve for the Global KDE algorithm.
    :return:
    """
    kde_df = read_kde()
    plot_roc_curve(kde_df, label="Global KDE")
    plt.show()

def plot_kalman_roc():
    """
    Plot the ROC curve for the Kalman filter algorithm.
    :return:
    """
    kalman_df = read_kalman()
    plot_roc_curve(kalman_df, label="Kalman filter")
    plt.show()

def plot_nn_roc():
    """
    Plot the ROC curve for the deep neural network algorithm.
    :return:
    """
    nn_df = read_nn_positions()
    plot_roc_curve(nn_df, label="Deep neural network")
    plt.show()

def plot_figure7(filename):
    """
    Plots the figure 7 of the paper.
    :return:
    """
    kdeamd = read_kdeamd(35, 35, 50)
    globalkde = read_kde()
    dmarkov = read_dmarkov(40, 40, 1, SymbolizationType.EQUAL_WIDTH, DivisionOrder.ROWS_THEN_COLUMNS)
    kalman = read_kalman()
    nn_positions = read_nn_positions()

    plot_roc_curves([kdeamd, globalkde, dmarkov, kalman, nn_positions],
                    labels=["KDE-AMD", "Global KDE", "$D$-Markov machine", "Kalman filter", "Deep neural network"])

    tikzplotlib.save(filename)

if __name__ == '__main__':
    # plot_figure7('figure7.tex')
    plot_kdeamd_roc()