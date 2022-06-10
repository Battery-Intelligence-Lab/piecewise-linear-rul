"""any code used for feature generation or selection"""
import numpy as np
import pandas as pd

from typing import List, Tuple


def form_datasets(
    featuredata: List[pd.DataFrame],
    cellnumbers: np.ndarray,
    ntraincells: int,
    nfeats: int,
    maxsim: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Take the data and return the desired form of the test and
    training data. This includes feature selection now. The outputs will be
    lists of the training and test data.
    """
    # index of first feature
    first_feature_index = 4
    # select the first numbers as the training set
    train_cells = cellnumbers[:ntraincells]
    # return test cell numbers
    test_cells = cellnumbers[ntraincells:]

    # # training set and target variable
    # init
    trainset0 = featuredata[train_cells[0]][
        featuredata[train_cells[0]].columns[first_feature_index:]
    ]
    y = featuredata[cellnumbers[0]][["dq"]]
    for j in range(1, ntraincells):
        # select full data
        f_j = featuredata[cellnumbers[j]]
        # reduce to just the input features
        f_j = f_j[f_j.columns[first_feature_index:]]
        # concatenate into larger array
        trainset0 = np.concatenate((trainset0, f_j))

        # target variable
        y = np.concatenate((y, featuredata[cellnumbers[j]][["dq"]]))

    # return the feature names
    trainset0 = pd.DataFrame(
        data=trainset0, columns=featuredata[cellnumbers[0]].columns[first_feature_index:]
    )

    # # Feature selection
    selected_features = feature_selection_func(trainset0, y, nfeats, maxsim)

    # form the list of data for the training set
    feats = ["t", "q", "dq"] + selected_features
    trainset = [featuredata[cell_id][feats].to_numpy() for cell_id in train_cells]

    # # test set
    feats = ["t", "q", "dq"] + selected_features
    testset = [featuredata[cell_id][feats].to_numpy() for cell_id in test_cells]

    return trainset, testset


def feature_selection_func(
    training_set: np.ndarray, target_vector: np.ndarray, nfeats: int, maxsim: float
):
    """
    Perform feature selection based on the pearsons rank.
    Trainset is a pandas dataframe, y is a column vector of
    the target variable. The output will be a list of columns
    of length nfeats. Maximum shared correlation is maxsim.
    """
    # concatenate data with target in indices 0
    data_array = np.hstack((training_set.to_numpy(), target_vector))

    # calculate coefficients
    similarity = np.abs(np.corrcoef(data_array, rowvar=False))
    # remove top row (don't want 'dq' to be returned)
    similarity = similarity[: similarity.shape[0] - 1, :]

    # return column names
    cols = training_set.columns

    # init selected_features
    sel_feats = []
    # loop through to find all the desired values
    for _ in range(nfeats):

        # find maximum correlation (column END)
        indx = np.argmax(similarity[:, similarity.shape[1] - 1])

        # return name
        sel_feats.append(cols[indx])

        # remove all necessary features
        keep_i = similarity[:, indx] < maxsim
        similarity = similarity[keep_i, :]
        similarity = similarity[:, np.hstack((keep_i, np.array([True])))]
        cols = cols[keep_i]

    return sel_feats


def form_training_array(X_list: List[np.ndarray]):
    """
    Form the training array by extracting the required data from
    the list of arrays

    Parameters
    ----------
    X_list : list of ndarrays
        training data

    Returns
    -------
    X_array : ndarray of the training data

    """
    # index of first feature
    first_feature_index = 3
    # required features
    feats = np.hstack((np.array(range(first_feature_index, X_list[0].shape[1])), np.array([2])))

    input_data_list = [cell_data[:, feats] for cell_data in X_list]

    X_array = np.vstack(input_data_list)

    return X_array
