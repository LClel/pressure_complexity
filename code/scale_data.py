from insole import *
import pickle as pk
import numpy as np
from project_insole_constants import *
from project_insole_analysis_information import *
from step_processing_functions import *
from all_data_processing_functions import *
import matplotlib.pyplot as plt


def scale_all_data(stomps):
    """ Collates data from all trials for all participants, scaled to participant 1 (size 11)
    Args:
        stomps: dictionary containing the time point at which the participant stomped in each trial
    Returns:
        data scaled to a uniform size
    """
    collated_data = {}

    file = '../preprocessed_data/PPT_001/trial07/left/left data.pbz2'
    data = decompress_pickle(file)

    target = data['Raw data']

    all_scaled = np.zeros((target.shape[0], target.shape[1], 0))
    participant_masses = []

    for foot in feet:
        print(foot)

        for participant in participant_ids:

            if calibration_type[participant] == 'point':
                continue
            else:
                print(participant)

                file = filepath_prefix + participant + '/trial00/' + foot + '/' + foot + ' data.pbz2'
                data = decompress_pickle(file)

                all_per_ppt = np.zeros((data['Raw data'].shape[0], data['Raw data'].shape[1], 0))

                for trial in trial_ids[5:-1]:
                    print(trial)

                    if participant not in details[trial]:
                        continue
                    else:
                        file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                        data = decompress_pickle(file)

                        stomp_start = stomps[trial][participant][0] + 5
                        stomp_end = stomps[trial][participant][1] - 5

                        if trial == 'trial05':
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, 1000:5500]))
                        elif trial == 'trial06':
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, 1000:4000]))
                        elif trial == 'trial10':
                            stand_end = trial_10_segmentation_times[participant][0]
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stand_end]))

                        elif trial == 'trial11':
                            stand_end = trial_11_segmentation_times[participant][0]
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stand_end]))
                        else:
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stomp_end]))

                # all_per_ppt = np.nan_to_num(all_per_ppt, nan=0.)
                scaled = scale_matrix(target, all_per_ppt)

                # create list of participant masses
                participant_masses.extend([participant_mass[participant]] * all_per_ppt.shape[2])

                all_scaled = np.dstack((all_scaled, scaled))

    for_calibration = np.zeros((all_scaled.shape[0], all_scaled.shape[1], 2))
    for_calibration[:, :, 0] = np.mean(all_scaled, axis=2)
    for_calibration[:, :, 1] = np.mean(all_scaled, axis=2)
    calibration_data = for_calibration

    D, min2, max2, min1, max1 = cut_frame(calibration_data, calibrate=True)

    s, regions, reshaped_data, idxs, D = map2footsim(D)

    collated_data['Raw data'] = all_scaled
    collated_data['Regions'] = regions
    collated_data['Reshaped data'] = reshaped_data
    collated_data['idxs'] = idxs
    collated_data['Stimulus'] = s
    collated_data['Calibration details'] = min2, max2, min1, max1

    compressed_pickle('../scaled_data/scaled raw data', collated_data)

    pk.dump(participant_masses, open("../scaled_data/ppt_masses.pkl", "wb"))


def scale_data_trial_type(stomps):
    """ Collates data from all trials for all participants, scaled to participant 1 (size 11)
        and split by trial type
    Args:
        stomps: dictionary containing the time point at which the participant stomped in each trial
    Returns:
        data scaled to a uniform size, saved in a dictionary with keys being trial type
    """

    # load all scaled all data to find sensor indexes within data and to ensure consistency across trials
    all_scaled_data = decompress_pickle('../scaled_data/scaled raw data.pbz2')
    all_scaled_raw = all_scaled_data['Raw data']
    all_scaled_idxs = all_scaled_data['idxs']
    min2, max2, min1, max1 = all_scaled_data['Calibration details']

    trial_type_classification = {'minimal movement': ['trial05', 'trial06', 'trial10', 'trial11', 'trial12'],
                                 'walking': ['trial07', 'trial08', 'trial09', 'trial13', 'trial14', 'trial15',
                                             'trial19'],
                                 'jumping': ['trial16', 'trial17', 'trial18']}

    target = all_scaled_raw
    collated_data = {}

    for trial_type in trial_type_classification:

        collated_data[trial_type] = {}

        all_scaled = np.zeros((target.shape[0], target.shape[1], 0))

        for foot in feet:

            for participant in participant_ids:

                if calibration_type[participant] == 'point':
                    continue
                else:
                    file = filepath_prefix + participant + '/trial00/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    all_per_ppt = np.zeros((data['Raw data'].shape[0], data['Raw data'].shape[1], 0))

                    for trial in trial_type_classification[trial_type]:

                        if participant not in details[trial]:
                            continue
                        else:
                            file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                            data = decompress_pickle(file)

                            stomp_start = stomps[trial][participant][0] + 5
                            stomp_end = stomps[trial][participant][1] - 5

                            if trial == 'trial05':
                                all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, 1000:5500]))
                            elif trial == 'trial06':
                                all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, 1000:4000]))
                            elif trial == 'trial10':
                                stand_end = trial_10_segmentation_times[participant][0]
                                all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stand_end]))

                            elif trial == 'trial11':
                                stand_end = trial_11_segmentation_times[participant][0]
                                all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stand_end]))
                            else:
                                all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stomp_end]))

                scaled = scale_matrix(target, all_per_ppt)

                all_scaled = np.append(all_scaled, scaled, axis=2)

        for_calibration = np.zeros((all_scaled.shape[0], all_scaled.shape[1], 2))
        for_calibration[:, :, 0] = np.mean(all_scaled, axis=2)
        for_calibration[:, :, 1] = np.mean(all_scaled, axis=2)
        calibration_data = for_calibration

        D = cut_frame(calibration_data, calibrate=False, min2=min2, max2=max2, min1=min1, max1=max1)

        s, regions, reshaped_data = map_given_locs(D, all_scaled_idxs)

        collated_data[trial_type]['Raw data'] = all_scaled
        collated_data[trial_type]['Regions'] = regions
        collated_data[trial_type]['Reshaped data'] = reshaped_data
        collated_data[trial_type]['idxs'] = all_scaled_idxs
        collated_data[trial_type]['Stimulus'] = s

    compressed_pickle('../scaled_data/scaled raw data by trial type', collated_data)


def scale_data_trials(stomps):
    """ Collates data from all trials for all participants, scaled to participant 1 (size 11)
        and split by trial
    Args:
        stomps: dictionary containing the time point at which the participant stomped in each trial
    Returns:
        data scaled to a uniform size, saved in a dictionary with keys being trial
    """

    # load all scaled all data to find sensor indexes within data and to ensure consistency across trials
    all_scaled_data = decompress_pickle('../scaled_data/scaled raw data.pbz2')
    all_scaled_raw = all_scaled_data['Raw data']
    all_scaled_idxs = all_scaled_data['idxs']
    min2, max2, min1, max1 = all_scaled_data['Calibration details']

    target = all_scaled_raw
    collated_data = {}

    for trial in trial_ids[5:-1]:
        print(trial)

        collated_data[trial] = {}

        all_scaled = np.zeros((target.shape[0], target.shape[1], 0))

        for foot in feet:
            print(foot)

            for participant in participant_ids:

                if calibration_type[participant] == 'point':
                    continue
                else:
                    print(participant)

                    file = filepath_prefix + participant + '/trial00/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    all_per_ppt = np.zeros((data['Raw data'].shape[0], data['Raw data'].shape[1], 0))

                    if participant not in details[trial]:
                        continue
                    else:
                        file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                        data = decompress_pickle(file)

                        stomp_start = stomps[trial][participant][0] + 5
                        stomp_end = stomps[trial][participant][1] - 5

                        if trial == 'trial05':
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, 1000:5500]))
                        elif trial == 'trial06':
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, 1000:4000]))
                        elif trial == 'trial10':
                            stand_end = trial_10_segmentation_times[participant][0]
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stand_end]))

                        elif trial == 'trial11':
                            stand_end = trial_11_segmentation_times[participant][0]
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stand_end]))
                        else:
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stomp_end]))

                all_per_ppt = np.nan_to_num(all_per_ppt, nan=0.)
                scaled = scale_matrix(target, all_per_ppt)

                scaled = (scaled - np.min(scaled)) / (np.max(scaled) - np.min(scaled))

                all_scaled = np.append(all_scaled, scaled, axis=2)

        for_calibration = np.zeros((all_scaled.shape[0], all_scaled.shape[1], 2))
        for_calibration[:, :, 0] = np.mean(all_scaled, axis=2)
        for_calibration[:, :, 1] = np.mean(all_scaled, axis=2)
        calibration_data = for_calibration

        D = cut_frame(calibration_data, calibrate=False, min2=min2, max2=max2, min1=min1, max1=max1)

        s, regions, reshaped_data = map_given_locs(D, all_scaled_idxs)

        collated_data[trial]['Raw data'] = all_scaled
        collated_data[trial]['Regions'] = regions
        collated_data[trial]['Reshaped data'] = reshaped_data
        collated_data[trial]['idxs'] = all_scaled_idxs
        collated_data[trial]['Stimulus'] = s

        # collated_data = collate_and_scale_data(filepath_prefix)

    compressed_pickle('../scaled_data/scaled raw data by trial', collated_data)


def scale_data_ppts(stomps):
    """ Collates data from all trials for all participants, scaled to participant 1 (size 11)
        and split by trial and participant
    Args:
        stomps: dictionary containing the time point at which the participant stomped in each trial
    Returns:
        data scaled to a uniform size, saved in a dictionary with keys being trial and participant
    """

    filepath_prefix = '/Volumes/My Passport/Project insole - official file w- restricted access/Data analysis/'
    # load all scaled all data to find sensor indexes within data and to ensure consistency across trials
    all_scaled_data = decompress_pickle('../scaled_data/scaled raw data.pbz2')
    all_scaled_raw = all_scaled_data['Raw data']
    all_scaled_idxs = all_scaled_data['idxs']
    min2, max2, min1, max1 = all_scaled_data['Calibration details']

    target = all_scaled_raw
    collated_data = {}

    for trial in trial_ids[5:-1]:

        collated_data[trial] = {}

        all_scaled = np.zeros((target.shape[0], target.shape[1], 0))

        for participant in participant_ids:

            collated_data[trial][participant] = {}

            if calibration_type[participant] == 'point':
                continue
            else:

                for foot in feet:

                    file = filepath_prefix + participant + '/trial00/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    all_per_ppt = np.zeros((data['Raw data'].shape[0], data['Raw data'].shape[1], 0))

                    if participant not in details[trial]:
                        continue
                    else:
                        file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                        data = decompress_pickle(file)

                        stomp_start = stomps[trial][participant][0] + 5
                        stomp_end = stomps[trial][participant][1] - 5

                        if trial == 'trial05':
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, 1000:5500]))
                        elif trial == 'trial06':
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, 1000:4000]))
                        elif trial == 'trial10':
                            stand_end = trial_10_segmentation_times[participant][0]
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stand_end]))

                        elif trial == 'trial11':
                            stand_end = trial_11_segmentation_times[participant][0]
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stand_end]))
                        else:
                            all_per_ppt = np.dstack((all_per_ppt, data['Raw data'][:, :, stomp_start:stomp_end]))

                    scaled = scale_matrix(target, all_per_ppt)

                    all_scaled = np.append(all_scaled, scaled, axis=2)

            for_calibration = np.zeros((all_scaled.shape[0], all_scaled.shape[1], 2))
            for_calibration[:, :, 0] = np.mean(all_scaled, axis=2)
            for_calibration[:, :, 1] = np.mean(all_scaled, axis=2)
            calibration_data = for_calibration

            D = cut_frame(calibration_data, calibrate=False, min2=min2, max2=max2, min1=min1, max1=max1)

            s, regions, reshaped_data = map_given_locs(D, all_scaled_idxs)

            collated_data[trial][participant]['Raw data'] = all_scaled
            collated_data[trial][participant]['Regions'] = regions
            collated_data[trial][participant]['Reshaped data'] = reshaped_data
            collated_data[trial][participant]['idxs'] = all_scaled_idxs
            collated_data[trial][participant]['Stimulus'] = s

        # collated_data = collate_and_scale_data(filepath_prefix)

    compressed_pickle('../scaled_data/scaled raw data by participant', collated_data)


def scaled_data_recalibration_constants():
    all_scaled_data = decompress_pickle('../scaled_data/scaled raw data.pbz2')
    all_scaled_raw = all_scaled_data['Raw data']
    all_scaled_idxs = all_scaled_data['idxs']
    min2, max2, min1, max1 = all_scaled_data['Calibration details']

    target = all_scaled_raw

    calibration_constants = {}

    for foot in feet:
        calibration_constants[foot] = {}

        for participant in participant_ids:

            if calibration_type[participant] == 'point':
                continue
            else:

                ### calculate calibration constant
                trial = 'trial01'
                file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                data = decompress_pickle(file)['Raw data']

                data = scale_matrix(target, data)

                # generaete calibration constants
                recalibrated, calibration_constant = recalibrate_data(data[:, :,
                                                                      separation_indexes['trial01'][
                                                                          participant][
                                                                          foot][0]:
                                                                      separation_indexes['trial01'][
                                                                          participant][
                                                                          foot][1]],
                                                                      mass=participant_mass['PPT_001'])

            calibration_constants[foot][participant] = calibration_constant

    compressed_pickle('../scaled_data/scaled_calibration_constants', calibration_constants)

    return calibration_constants