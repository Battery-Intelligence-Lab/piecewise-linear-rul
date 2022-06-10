"""code for calculate preformance"""
import numpy as np

from typing import List

from numpy import matmul as mm
from numpy.linalg import inv
from numpy import percentile
from scipy.interpolate import UnivariateSpline as Spline
from scipy.stats import norm

from soh_prognosis_plr import blr, blrmean

EOL_SOH = 80.0


def calculate_results(results_list: List[np.ndarray], eol_soh: float = EOL_SOH):
    """
    Takes the results list and returns all the values that need to be stored
    into the result_class object

    Need RMSE cap, EoL error, prctiles, obs_freq and rmse_freq
    """
    R = result_class()

    R_list = results_list

    ######################
    #### RMSE and EoL ####
    ######################

    # data storage
    rmse = np.zeros((len(R_list), 1))
    eols = np.zeros((len(R_list), 1))
    for j in range(len(R_list)):
        # select data
        f = R_list[j]
        n = f.shape[0]
        # columns = [time, obs_cap, pred_cap, pred_std]

        # create data columns
        t = f[:, 0].reshape((n, 1))
        qm = f[:, 1].reshape((n, 1))
        qp = f[:, 2].reshape((n, 1))
        # qs = f[:,3].reshape((n,1))

        # RMSE
        rmse[j, 0] = np.sqrt(np.mean((qm - qp) ** 2))

        # EOL
        # change the eol health if observed capacity does
        # not reach eol_soh
        eol_j = np.maximum(eol_soh, qm.min())

        # calculate eols
        eol_m = eol_calculator(t, qm, eol_j)
        eol_p = eol_calculator(t, qp, eol_j)

        # return percentage error
        eols[j, 0] = abs((1 - (eol_p / eol_m)) * 100)

    # remove any nans (usually a very, very small number)
    eols = eols[np.isfinite(eols)]

    # percentiles
    R.rmse_cap = np.array([percentile(rmse, 50), percentile(rmse, 95)])
    R.eol_error = np.array([percentile(eols, 50), percentile(eols, 95)])

    #####################
    #### Frequencies ####
    #####################

    # form large array
    r = results_list[0]
    for j in range(1, len(R_list)):
        r = np.concatenate((r, R_list[j]))
    # columns = [time, obs_cap, pred_cap, pred_std]

    # return the cumulative probability of where the observation
    # lies with a mean of the prediction and standard devation
    # as output. In units of %
    r_cdf = norm.cdf(r[:, 1], r[:, 2], r[:, 3]).reshape((r.shape[0], 1)) * 100

    # define percentiles and store it
    prcts = np.linspace(1, 99, 99).reshape((99, 1))
    R.prcts = prcts

    # boolean array of above/below for False/True by data point
    # percentage
    M = r_cdf < prcts.T

    # calculate percentage frequencies
    f = np.sum(M, axis=0).reshape((M.shape[1], 1)) * 100 / r.shape[0]

    # store
    R.obs_freq = f

    # calc and store the rmse freq
    rms_freq = np.sqrt(np.mean((f - prcts) ** 2))

    R.rmse_freq = rms_freq

    return R


class result_class:
    # for storing the results of each stage
    def n(self):
        # number of points in training set (integer)
        return self

    def prcts(self):
        # percentage frequencies (vector)
        return self

    def rmse_freq(self):
        # rmse of the frequencies (double)
        return self

    def obs_freq(self):
        # observed frequencies (vector)
        # same size as prcts
        return self

    def rmse_cap(self):
        # median and 95th percentiles of the rmse
        return self

    def eol_error(self):
        # median and 95th percentiles of eol error
        return self


def eol_calculator(t, q, eol_soh):
    """
    takes a capacity profile and calculates when it will
    cross the eol_soh value.
    """
    if q.min() <= eol_soh:
        # form new indep variable spanning full length
        x = np.linspace(t.min(), t.max(), 1000).reshape((1000, 1))
        # spline fit of capacity-time profile
        spl_q = Spline(t, q)
        # new capacity profile
        y = spl_q(x)
        # find index of end of life (closest)
        indx_eol = np.argmin(np.abs(y - eol_soh))
        # find time
        t_eol = x[indx_eol]

    else:
        # linear extrapolation based on last 3 data points
        n = t.shape[0]
        k = 3
        t = t[n - k : n].reshape((k, 1))
        q = q[n - k : n].reshape((k, 1))

        w = blr(q, t, 1, np.eye(2) * (100**2))
        t_eol = blrmean(np.array([[eol_soh]]), w)

    return t_eol


def disp_profile_results(trialname, results):
    """
    display profile results
    """
    print("=== " + trialname + " ===")
    print("   RMSE Capacity: ")
    print("             Median = " + str(np.round(results.rmse_cap[0], 2)) + "%")
    print("               95th = " + str(np.round(results.rmse_cap[1], 2)) + "%")
    print("  ")
    print("   EoL Error: ")
    print("             Median = " + str(np.round(results.eol_error[0], 2)) + "%")
    print("               95th = " + str(np.round(results.eol_error[1], 2)) + "%")
    print("  ")
