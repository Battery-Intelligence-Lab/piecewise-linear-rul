"""code for piecewise lineaar model"""
import numpy as np
import pandas as pd

from numpy import matmul as mm
from numpy.linalg import inv
from numpy import percentile
from scipy import signal

from typing import List, Tuple

from feature_engineering import form_datasets, form_training_array


def fixdata(featuredata: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Input a dataframe created by reading the features csv and produce a
    list of the data sorted by cell number.
    """
    list_of_feature_data = []
    for cell_id in featuredata["cellID"].unique():

        indx = np.where(featuredata["cellID"].to_numpy() == cell_id)[0]

        list_of_feature_data.append(featuredata.iloc[indx, :][featuredata.columns[1:]])

    return list_of_feature_data


def generate_cell_numbers(number_of_cells: int, repeats: int = 10) -> List[np.ndarray]:
    """
    Produces the cell number selections for a trial based on
    the total number of cells
    """
    return [np.random.permutation(range(number_of_cells)) for _ in range(repeats)]


def piecewise_linear_model_wrapper(
    input_feature_data: List[pd.DataFrame],
    cell_numbers: np.ndarray,
    ntraincells: int = 50,
    ninputfeats: int = 5,
    rho_p_max: float = 0.85,
    beta_L: float = 0.1,
    beta_improv: float = 0.01,
    max_mods: int = 10,
    sigma_w_0: float = 10,
) -> List[np.ndarray]:
    """
    Wrapper piecewise linear regression model

    Parameters
    ----------

    Returns
    -------
    resultd_test : list
        List of result arrays, each with columns of
        [time, capacity, predicted capacity, prediction standard deviation].

    """

    # load train and test sets after Feature Selection
    trainset, testset = form_datasets(
        input_feature_data,
        cell_numbers,
        ntraincells,
        ninputfeats,
        rho_p_max,
    )

    # run piecewise linear model
    results_test = piecewise_linear_function(
        trainset, testset, beta_L, beta_improv, max_mods, sigma_w_0
    )

    return results_test


def costfunc(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    computes the cost function for a given fit

    Currently set up to calculate RMSE between input
    vectors, y1 and y2.
    """
    cost = np.sqrt(np.mean((y1 - y2) ** 2))  # RMSE
    return cost


def blr(x: np.ndarray, y: np.ndarray, b2: float, S: np.ndarray) -> np.ndarray:
    """
    function will produce the coefficients for a bayesian
    linear regression. function automatically adds the
    shift column

    b2 is the precision (1/\sigma_n^2)
    S is the covariance form the prior over parameters
    x and y need to be 2D arrays of the same height.

    output w will be a column vector of the coefficients
    """
    # add the column of ones as the bias
    X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    # full calculation
    w = b2 * (mm(inv(b2 * mm(X.T, X) + inv(S)), mm(X.T, y)))

    # return w
    return w


def blrmean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    function takes parameters w and fits the mean function
    """
    X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    y = mm(X, w)
    return y


def A_blr(X: np.ndarray, sigma_n_2: float, sigma_w: np.ndarray) -> np.ndarray:
    # add bias term
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # calculate the inverse of the posterior covariance
    a = mm(X.T, X) / sigma_n_2 + inv(sigma_w)
    return a


def var_blr(Xt: np.ndarray, X: np.ndarray, sigma_n_2: float, sigma_w: np.ndarray) -> np.ndarray:
    # add bias term
    Xt = np.concatenate((np.ones((Xt.shape[0], 1)), Xt), axis=1)
    # calculate variance
    v = mm(Xt, mm(inv(A_blr(X, sigma_n_2, sigma_w)), Xt.T))
    return v


class BLR_Distribution:
    def sigma_n_2(self):  # observation noise
        return self

    def sigma_w(self):  # prior covariance
        return self

    def w(self):  # estimated coefficients
        return self


def piecewise_linear_function(
    trainset: List[np.ndarray],
    testset: List[np.ndarray],
    beta_L: float,
    beta_improv: float,
    maximum_models: int,
    sigma_w_0: float,
) -> List[np.ndarray]:
    """
    Performs the actual lifetime modelling and returns the required
    values. This is a mjor function. Before I go anywhere, I need
    to make sure that all of the other functions are in place
    """
    # form training set
    trainset_array = form_training_array(trainset)

    nf = trainset_array.shape[1] - 1

    # training X and y
    y_train = trainset_array[:, -1].reshape((trainset_array.shape[0], 1))
    X_train = trainset_array[:, :-1].reshape((trainset_array.shape[0], nf))

    # cost vector
    cost = np.zeros((0, 2))

    # train initial model
    initmods = 1
    bpts0, cost0, w0, dists0 = piecewise_model_func(X_train, y_train, initmods, sigma_w_0, beta_L)
    cost = np.vstack((cost, np.array([[initmods, cost0]])))

    # loop up to maximum number of allowed models
    for nmods in range(initmods + 1, maximum_models):

        # train model
        bpts1, cost1, w1, dists1 = piecewise_model_func(X_train, y_train, nmods, sigma_w_0, beta_L)

        cost = np.vstack((cost, np.array([[nmods, cost1]])))

        # store outputs if the above loop allows continuaton
        w0 = w1
        bpts0 = bpts1
        cost0 = cost1
        dists0 = dists1

    # select model size
    indx = np.argmin(cost[:, 1])
    costmin = cost[indx, 1] * (1 + beta_improv)
    cost[cost[:, 1] > costmin, 0] = maximum_models
    indx = np.argmin(cost[:, 0])
    nmods = cost[indx, 0].astype(int)

    # re-train model (easier than storing them all, sorry)
    bpts0, cost0, w0, dists0 = piecewise_model_func(X_train, y_train, nmods, sigma_w_0, beta_L)

    # save outputs
    w = w0
    bpts = bpts0
    dists = dists0

    # training results
    results_test = plr_results_calculation(testset, w, bpts, dists, X_train)

    return results_test


### FUNCTION TO PRODUCE PIECEWISE OUTPUTS ###
def produce_piecewise_output(X: np.ndarray, w: np.ndarray, bpts: np.ndarray) -> np.ndarray:
    """
    Function calculates the required output for input values X
    with parameters w and breakpoints bpts.
    """
    # number of models
    nmodels = len(w)
    # store splitting variable
    x0 = X[:, 0].reshape((X.shape[0], 1))

    # initialise output
    yp = np.zeros((X.shape[0], 1))
    for j in range(nmodels):
        # range of variables
        xmin = bpts[0, j]
        xmax = bpts[0, j + 1]

        # count data points
        nj = np.sum(np.array((x0 > xmin) & (x0 <= xmax)))

        # only proceed IF there are data points
        if nj >= 1:
            # select relevant index
            indx = np.where((x0 > xmin) & (x0 <= xmax))[0]
            # select test inputs
            Xj = X[indx, :].reshape((nj, X.shape[1]))
            # produce output
            yp_j = blrmean(Xj, w[j])
            # put output into output vector
            yp[indx, :] = yp_j

    # return the output
    return yp


def plr_results_calculation(
    testdata: List[np.ndarray],
    w: np.ndarray,
    bpts: np.ndarray,
    dists: BLR_Distribution,
    X_train: np.ndarray,
) -> List[np.ndarray]:
    """
    Produce the output data based on a test dataset and a plr model. The model
    will be a dict type.
    """
    # initialise results stuff
    Results = []

    # predict the test outputs
    for j in range(len(testdata)):
        # select test data
        n = testdata[j].shape
        X_test = testdata[j][:, 3 : n[1]].reshape((n[0], n[1] - 3))

        # produce predictions
        dyp = produce_piecewise_output(X_test, w, bpts)

        # convert to capacity
        yp = testdata[j][0, 1] + np.cumsum(dyp).reshape(dyp.shape)

        # return complex
        X_res = np.concatenate((testdata[j][:, [0, 1]], yp), axis=1)

        # append
        Results.append(X_res)

    ### UNCERTAINTY SECTION
    # this only uses the standard output for a BLR process.
    # no sparse/approx/other techniques
    sigma_n_2 = dists.sigma_n_2
    sigma_w = dists.sigma_w
    for j in range(len(testdata)):
        # select test data
        n = testdata[j].shape
        X_test = testdata[j][:, 3 : n[1]].reshape((n[0], n[1] - 3))

        # splitting variables
        xt0 = X_test[:, 0].reshape((X_test.shape[0], 1))
        x0 = X_train[:, 0].reshape((X_train.shape[0], 1))

        # init output
        var_dyp = np.zeros(xt0.shape)

        for jj in range(bpts.shape[1] - 1):
            # variable bounds from breakpoints
            xmin = bpts[0, jj]
            xmax = bpts[0, jj + 1]

            # sum points in breakpoint range
            nt_jj = np.sum(np.array((xt0 > xmin) & (xt0 <= xmax)))
            n_jj = np.sum(np.array((x0 > xmin) & (x0 <= xmax)))

            if nt_jj > 0:
                # select data
                indx_t = np.where((xt0 > xmin) & (xt0 <= xmax))[0]
                Xt_jj = X_test[indx_t, :].reshape((nt_jj, X_test.shape[1]))

                indx = np.where((x0 > xmin) & (x0 <= xmax))[0]
                X_jj = X_train[indx, :].reshape((n_jj, X_train.shape[1]))

                # variance
                V = var_blr(Xt_jj, X_jj, sigma_n_2, sigma_w)

                # select the diagonal elements
                var_jj = np.diag(V).reshape((nt_jj, 1))

                # return variance values and combine with observation noise
                var_dyp[indx_t, :] = var_jj + sigma_n_2

        # transform to capacity
        std_yp = np.sqrt(np.cumsum(var_dyp).reshape((xt0.shape)))

        # form results array
        X_res = np.concatenate((Results[j], std_yp), axis=1)

        # final return
        Results[j] = X_res

    return Results


# weighted moving average
def wma(
    splitting_variable: np.ndarray, x0: np.ndarray, y0: np.ndarray, beta_L: float
) -> np.ndarray:
    """
    Gaussian moving mean with lengthscale set by x0
    """
    # lengthscale
    sigma_l = (percentile(x0, 99) - percentile(x0, 1)) * beta_L
    # weights
    rbf = np.exp(-((splitting_variable - x0.T) ** 2) / (sigma_l**2))
    # ys
    Y = y0.T * np.ones(splitting_variable.shape)
    # normalisation factor
    N = np.sum(rbf, axis=1).reshape((rbf.shape[0], 1)) * np.ones((1, rbf.shape[1]))
    # weighted moveing average
    y1 = np.sum(Y * rbf / N, axis=1)
    return y1


### SPLITTING FUNCTION UP USING CURVATURE
def splitting_by_curvature(
    splitting_variable: np.ndarray,
    target_variable: np.ndarray,
    nbps: int,
    minpts: int,
    beta_L: float,
) -> Tuple[np.ndarray, int]:
    """
    Produces the splits in the piecewise model.

    splitting_variable is the splitting feature raw data
    """
    # data points
    datapoints = splitting_variable.shape[0]

    # sort data according to values of splitting_variable
    indx = np.argsort(splitting_variable, axis=0)
    splitting_variable = splitting_variable[indx].reshape((splitting_variable.shape[0], 1))
    target_variable = target_variable[indx].reshape((splitting_variable.shape[0], 1))

    # select fitting and targt data, eliminating the extremes
    splitting_variable = splitting_variable[minpts:-minpts, 0]
    target_variable = target_variable[minpts:-minpts, 0]

    datapoints = splitting_variable.shape[0]

    splitting_variable = splitting_variable.reshape((datapoints, 1))
    target_variable = target_variable.reshape((splitting_variable.shape[0], 1))

    # smoothed function
    x = np.linspace(splitting_variable.min(), splitting_variable.max(), 1000).reshape((1000, 1))
    f = wma(x, splitting_variable, target_variable, beta_L)

    # first derivative
    dfdx = np.diff(f) / np.diff(x.squeeze())

    # second derivative
    df2dx2 = np.abs(np.diff(dfdx) / np.diff(x.squeeze())[1:])
    df2dx2 = df2dx2.reshape(df2dx2.shape[0], 1)

    # population weighting
    d = (percentile(splitting_variable, 99) - percentile(splitting_variable, 1)) * 0.1
    D = np.abs(x[2:] - splitting_variable.T)
    # population density func
    rho = np.sum(D < d, axis=1)

    # smooth population density func
    rho_s = signal.savgol_filter(rho / np.mean(rho), 51, 3)

    # ============================================================
    # This set up uses the multiplied function of y and the
    # population density The overall coding is simpler, even if
    # the actual function isn't
    F_s = rho_s.reshape((rho_s.shape[0], 1)) * df2dx2

    # find peaks in combined function
    peak_indx = signal.find_peaks(F_s.reshape((F_s.shape[0],)))[0]
    peak_nums = np.flip(np.argsort(F_s[peak_indx, 0]))

    # return desired breakpoints
    if nbps > 0:
        npts = np.min([nbps, peak_nums.shape[0]])
        bpts = x[peak_indx[peak_nums[0:nbps]] + 2].reshape((1, npts))
        # sort the final values
        bpts = np.sort(bpts)
    else:
        bpts = np.array([[]])  # return empty if no breakpoints desired

    # count possible number of breakpoints
    maxbpts = peak_indx.shape[0]

    # and there go the values
    return bpts, maxbpts


### PIECEWISE MODEL FUNCTION
def piecewise_model_func(
    Xtrainset: np.ndarray,
    target: np.ndarray,
    nmodels: int,
    sigma_w_0: float,
    beta_L: float,
) -> Tuple[np.ndarray, float, List[np.ndarray], BLR_Distribution]:
    """
    train a piecewise linear model using known training data and parameters
    """
    # controls/priors
    minpts = 100

    # sigma_w_0 = M['sigma_w_0'] TODO
    sigma_w = np.eye(Xtrainset.shape[1] + 1) * (sigma_w_0**2)

    # variable changes
    X = Xtrainset
    y = target
    xshape = X.shape

    # approxiamte measurement noise using a weighted moving average
    x0 = X[:, 0].reshape((xshape[0], 1))

    # estimate observation noise based on a smoothed function
    beta_L_y = 0.1
    y_wma = wma(x0, x0, y, beta_L_y)
    sigma_n_2 = np.mean((y_wma - y) ** 2)  # approximate observation noise

    sigma_n_2 = sigma_n_2 / 2

    # estimated precision
    beta_2 = sigma_n_2**-1

    # calculate break points
    bpts, maxbpts = splitting_by_curvature(x0, y, nmodels - 1, minpts, beta_L)

    # concatenate with infinities
    bpts = np.concatenate(
        (np.array(-np.inf).reshape((1, 1)), bpts, np.array(np.inf).reshape((1, 1))), axis=1
    )

    # store bayesian priors and final parameters
    dist = BLR_Distribution()
    dist.sigma_n_2 = sigma_n_2
    dist.sigma_w = sigma_w

    # store model if model of size can be produced
    if maxbpts + 1 >= nmodels:
        w = []  # initiate list of parameters
        yp = np.zeros(y.shape)

        for j in range(nmodels):
            # range of variables
            xmin = bpts[0, j]
            xmax = bpts[0, j + 1]

            # count number of data points in sub-model
            nj = sum((x0 > xmin) & (x0 <= xmax))[0]

            # isolate data for sub-model
            indx = np.where((x0 > xmin) & (x0 <= xmax))[0]
            Xj = X[indx, :].reshape((nj, xshape[1]))
            yj = y[indx, :].reshape((nj, 1))

            # calculate the coefficients
            w_j = blr(Xj, yj, beta_2, sigma_w)
            w.append(w_j)

            # produce training set output for sub-model
            indx = np.where((x0 > xmin) & (x0 <= xmax))[0]
            yp[indx, :] = blrmean(Xj, w_j)

        # calculate cost function
        cost = costfunc(y, yp)

        # store params
        dist.w = w

        # new σ_n_2
        # This update has a marked improvement on the credible
        # intervals (obviously has no impact of mean function)
        dist.sigma_n_2 = np.array((y - yp)).reshape(y.shape).var()

    else:
        cost = np.inf
        w = []

    return bpts, cost, w, dist
