import numpy as np
from sklearn.decomposition import NMF, non_negative_factorization
import matplotlib.pyplot as plt
import pandas as pd
import bz2
import _pickle as cPickle
import seaborn as sns
from insole import *
from project_insole_constants import *

def explained_variance(true, W, H):
    """ Calculate explained variance

    Args:
        true (np.array()): true data
        W (np.array()): NMF weight matrix
        H (np.array()): NMF component matrix

    Returns:
        explained variance (float) between 0 and 1


    """

    return 1 - np.var(true - W.dot(H)) / np.var(true)


def nmf_all_trials(scaled):
    """ Conduct NMF on each individual trial
    Args:
        scaled (dict containing np.array()): data for a single trial from all participants scaled to a uniform size
    Returns:
        Saves 2 dictionaries:
            nmf_dfs: explained variance by each NMF component for each trial
            nmf_weights: NMF weight and component matrices for each trial
    """
    nmf_dfs = {}
    nmf_weights = {}

    # loop through each trial
    for trial in scaled:

        # create list of numbers 1 to 30
        components = list(range(1, 31))


        D = scaled[trial]['Raw data']
        idxs = scaled[trial]['idxs']
        reshaped = np.zeros((D.shape[0] * D.shape[1], D.shape[2]))
        k = 0
        for j in range(D.shape[1]):
            for i in range(D.shape[0]):
                reshaped[k] = D[i, j, :]

                k += 1

        # remove the empty data
        data = reshaped[idxs]
        # ensure that any nans are changed to zero
        data = np.nan_to_num(data)

        data = data.T

        perfs = []

        # run NMF on the data with components 1 to 30
        for i in range(len(components)):
            nmf = NMF(n_components=components[i])
            W = nmf.fit_transform(data)
            H = nmf.components_

            # calculate variance explained
            perfs.append(explained_variance(data, W, H))

        nmf_dfs[trial] = pd.DataFrame({'Component': list(range(30)), 'Variance explained': pd.Series(perfs)})

        nmf_weights[trial] = {}
        nmf_weights[trial]['H'] = H
        nmf_weights[trial]['W'] = W

    # save data
    compressed_pickle('../processed_data/nmf_dfs', nmf_dfs)

    compressed_pickle('../processed_data/nmf_weights', nmf_weights)



def nmf_trial_type(scaled):
    """ Conduct NMF on each individual trial

    Args:
        scaled (dict containing np.array()): data for each trial type from all participants scaled to a uniform size

    Returns:
        Saves 2 dictionaries:
            nmf_dfs: explained variance by each NMF component for each trial type
            nmf_weights: NMF weight and component matrices for each trial type

    """
    nmf_dfs = {}
    nmf_weights = {}
    for trial in scaled:

        # create list of numbers 1 to 30
        components = list(range(1, 31))

        D = scaled[trial]['Raw data']
        idxs = scaled[trial]['idxs']
        reshaped = np.zeros((D.shape[0] * D.shape[1], D.shape[2]))
        k = 0
        for j in range(D.shape[1]):
            for i in range(D.shape[0]):
                reshaped[k] = D[i, j, :]

                k += 1

        # remove the empty data
        data = reshaped[idxs]
        # ensure that any nans are changed to zero
        data = np.nan_to_num(data)

        data = data.T

        perfs = []

        for i in range(len(components)):
            nmf = NMF(n_components=components[i])
            W = nmf.fit_transform(data)
            H = nmf.components_

            # calculate variance explained
            perfs.append(explained_variance(data, W, H))

        nmf_dfs[trial] = pd.DataFrame({'Component': list(range(30)), 'Variance explained': pd.Series(perfs)})

        nmf_weights[trial] = {}
        nmf_weights[trial]['H'] = H
        nmf_weights[trial]['W'] = W

    # save data
    compressed_pickle('../processed_data/trial_type_nmf_dfs', nmf_dfs)

    compressed_pickle('../processed_data/trial_type_nmf_weights', nmf_weights)


def inverse_transform_trials(dfs, models, scaled):
    """ Calculate new NMF weights while training on fixed components - uses the first 5 components from each trial

    Args:
        dfs (dict containing pd.DataFrame()): variance explained by each component
        models (dict containing np.array()): contains NMF component matrices from NMF on each trial
        scaled (dict containing np.array()): data for a single trial from all participants scaled to a uniform size

    Returns:
        figure of variance explained on each trial

    """
    trial_ids = ['trial05', 'trial06', 'trial12', 'trial10', 'trial11', 'trial09', 'trial07', 'trial08', \
                 'trial15', 'trial13', 'trial14', 'trial19', 'trial16', 'trial17', 'trial18']
    all_trial_titles = ['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk', 'Slow walking', 'Normal walking', 'Fast walking', \
                        'Jogging', 'Slope walking', 'Stairs', 'Gravel walking', 'Both feet jump', 'Left foot jump', 'Right foot jump']

    var_exp = np.zeros((len(trial_ids), len(trial_ids)))
    # loop through trials
    for i in range(len(trial_ids)):

        H = models[trial_ids[i]]['H']

        li = np.asarray(dfs[trial_ids[i]]['Component'])

        H_new = np.zeros((0, H.shape[1]))

        # extract the first 5 components explaining the greatest variance
        for n in range(5):
            H_new = np.vstack((H_new, H[li[n], :]))

        for k in range(len(trial_ids)):

            D = scaled[trial_ids[k]]['Raw data']
            idxs = scaled[trial_ids[k]]['idxs']
            reshaped = np.zeros((D.shape[0] * D.shape[1], D.shape[2]))
            m = 0
            for q in range(D.shape[1]):
                for l in range(D.shape[0]):
                    reshaped[m] = D[l, q, :]

                    m += 1

            # remove the empty data
            data = reshaped[idxs]
            # ensure that any nans are changed to zero
            data = np.nan_to_num(data)

            data = data.T

            # train NMF model using fixed weights
            Wtest, Htest, n_iter = non_negative_factorization(data, H=H_new, n_components=5, update_H=False)

            var_exp[i, k] = explained_variance(data, Wtest, H_new)

    plt.figure(figsize=(10, 10), dpi=600)
    sns.heatmap(var_exp, xticklabels=all_trial_titles, yticklabels=all_trial_titles, annot=True, \
                square=True, cmap='plasma', vmin=0, vmax=1, fmt='.2f', cbar=False)
    plt.tight_layout()
    plt.savefig('../individual_figures/all_nmf_inverse_heatmap_5.png',dpi=600)


def inverse_transform_trial_type(trial_type_dfs, trial_type_models, scaled):
    """ Calculate new NMF weights while training on fixed components - uses the first 5 components from each trial type

        Args:
            dfs (dict containing pd.DataFrame()): variance explained by each component
            models (dict containing np.array()): contains NMF component matrices from NMF on each trial type
            scaled (dict containing np.array()): data for a single trial type from all participants scaled to a uniform size

        Returns:
            figure of variance explained on each trial using trial type components

        """

    sorted_trial_ids = ['trial05', 'trial06', 'trial12', 'trial10', 'trial11', 'trial09', 'trial07', 'trial08', \
                 'trial15', 'trial13', 'trial14', 'trial19', 'trial16', 'trial17', 'trial18']
    sorted_trial_titles = ['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk', 'Slow walking', 'Normal walking', 'Fast walking', \
                        'Jogging', 'Slope walking', 'Stairs', 'Gravel walking', 'Both feet jump', 'Left foot jump', 'Right foot jump']

    old_trial_types = list(trial_type_dfs.keys())
    type_var_exp = np.zeros((len(old_trial_types), len(sorted_trial_ids)))

    # loop through trial types
    for i in range(len(old_trial_types)):

        H = trial_type_models[old_trial_types[i]]['H']

        li = np.asarray(trial_type_dfs[old_trial_types[i]]['Component'])

        H_new = np.zeros((0, H.shape[1]))

        # extract the first 5 NMF components
        for n in range(5):

            H_new = np.vstack((H_new, H[li[n], :]))

        for k in range(len(sorted_trial_ids)):

            D = scaled[sorted_trial_ids[k]]['Raw data']
            idxs = scaled[sorted_trial_ids[k]]['idxs']
            reshaped = np.zeros((D.shape[0] * D.shape[1], D.shape[2]))
            m = 0
            for q in range(D.shape[1]):
                for l in range(D.shape[0]):
                    reshaped[m] = D[l, q, :]

                    m += 1

            # remove the empty data
            data = reshaped[idxs]
            # ensure that any nans are changed to zero
            data = np.nan_to_num(data)

            data = data.T

            # run NMF on fixed components
            Wtest, Htest, n_iter = non_negative_factorization(data, H=H_new, n_components=5, update_H=False)

            # calculate variance explained
            type_var_exp[i, k] = explained_variance(data, Wtest, H_new)

    plt.figure(figsize=(10, 10), dpi=600)
    sns.heatmap(type_var_exp, xticklabels=sorted_trial_titles, yticklabels=trial_types, \
                annot=True, square=True, cmap='plasma', vmin=0, vmax=1, fmt='.2f', cbar=False)
    plt.tight_layout()
    plt.savefig(
        '../individual_figures/type_all_nmf_inverse_heatmap_5.png',
        dpi=600)
