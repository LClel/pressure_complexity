from insole import *
import pickle as pk
import numpy as np
from project_insole_constants import *


def data_collation_function():
    """ Reads in raw data and runs preprocessing on the data. Also calculates and saves calibration constants

    Returns:

    """
    prefix = '../raw_data/'

    output_path_prefix = '../preprocessed_data/'

    calibration_constants = {}
    participant_stimulus_indexes = {}

    # loop through feet
    for foot in feet:
        print(foot)
        calibration_constants[foot] = {}

        participant_stimulus_indexes[foot] = {}

        # define file suffix depending on foot
        if foot == 'left':
            suffix = 'L_M'

        else:
            suffix = 'R_M'

        # loop through participants
        for participant in participant_ids:
            print(participant)
            all_trials = np.zeros((59, 21, 1))

            # loop through all trial IDs
            for trial in trial_ids:
                print(trial)

                if calibration_type[participant] == 'point':
                    continue
                else:

                    # checks if participant took part in the trial
                    if participant not in details[trial]:
                        continue
                    else:
                        # find trial number to use in filepath
                        trial_number = trial[-2:]

                        filepath = prefix + participant + '/' + trial + '/' + dates[participant] + 'PPT' + str(
                            trial_number) + suffix + '.csv'

                        # read in raw data
                        data = import_data(filepath=filepath, frames=frames_per_trial[participant][trial],
                                           calibration_type=calibration_type[participant],
                                           extended_calibration=extended_calibration[participant][trial])

                        # filter signal
                        data = filter_data(data)

                        all_trials = np.append(all_trials, data, axis=2)

            else:
                trial_number = trial[-2:]

                filepath = prefix + participant + '/' + trial + '/' + dates[participant] + 'PPT' + str(trial_number) + suffix + '.csv'

                data = import_data(filepath=filepath, frames=frames_per_trial[participant][trial], calibration_type=calibration_type[participant], extended_calibration=extended_calibration[participant][trial])

                # filter signal
                data = filter_data(data)

                all_trials = np.append(all_trials, data, axis=2)

            if foot == 'right':
                all_trials = flip_matrix(all_trials)

            # concatenate all data
            calibration_data = create_calibration_data(all_trials, all_trials)

            # generate data to calibrate cut frame
            for_calibration = np.zeros((calibration_data.shape[0], calibration_data.shape[1], 2))
            for_calibration[:, :, 0] = np.mean(calibration_data, axis=2)
            for_calibration[:, :, 1] = np.mean(calibration_data, axis=2)
            calibration_data = for_calibration

            # find cut frame indexes
            D, min2, max2, min1, max1 = cut_frame(calibration_data, calibrate=True)

            # generate common FootSim stimulus object
            s, regions, reshaped_data, idxs, D = map2footsim(D)

            participant_stimulus_indexes[foot][participant] = idxs



            ### calculate calibration constant
            trial = 'trial01'
            trial_number = trial[-2:]
            filepath = prefix + participant + '/' + trial + '/' + dates[participant] + 'PPT' + str(
                trial_number) + suffix + '.csv'

            data = import_data(filepath=filepath, frames=frames_per_trial[participant][trial],
                               calibration_type=calibration_type[participant],
                               extended_calibration=extended_calibration[participant][trial])

            # filter signal
            data = filter_data(data)

            if foot == 'right':
                data = flip_matrix(data)

            # cut data to uniform size per participant
            cut_data = cut_frame(data, calibrate=False, min2=min2, max2=max2, min1=min1, max1=max1)

            # generaete calibration constants
            recalibrated, calibration_constant = recalibrate_data(cut_data[:, :,
                                                                  separation_indexes['trial01'][
                                                                      participant][
                                                                      foot][0]:
                                                                  separation_indexes['trial01'][
                                                                      participant][
                                                                      foot][1]],
                                                                  mass=participant_mass[participant])


            calibration_constants[foot][participant] = calibration_constant


            for trial in trial_ids:


                if calibration_type[participant] == 'point':
                    continue
                else:


                    if participant not in details[trial]:
                        continue
                    else:
                        trial_number = trial[-2:]

                        filepath = prefix + participant + '/' + trial + '/' + dates[participant] + 'PPT' + str(
                            trial_number) + suffix + '.csv'

                        data = import_data(filepath=filepath, frames=frames_per_trial[participant][trial],
                                           calibration_type=calibration_type[participant],
                                           extended_calibration=extended_calibration[participant][trial])

                        data = data * calibration_constants[foot][participant]

                        # filter signal
                        data = filter_data(data)

                        if foot == 'right':
                            data = flip_matrix(data)

                        output_path = output_path_prefix + participant + '/' + trial + '/' + foot + '/'

                        cut_data = cut_frame(data, calibrate=False, min2=min2, max2=max2, min1=min1, max1=max1)

                        data = all_frames_analysis(cut_data, participant, foot, frames_per_trial[participant][trial], idxs,
                                                   output_path=output_path,plot=False)


            else:
                trial_number = trial[-2:]

                filepath = prefix + participant + '/' + trial + '/' + dates[participant] + 'PPT' + str(trial_number) + suffix + '.csv'

                data = import_data(filepath=filepath, frames=frames_per_trial[participant][trial],
                                   calibration_type=calibration_type[participant],
                                   extended_calibration=extended_calibration[participant][trial])

                data = data * calibration_constants[foot][participant]

                # filter signal
                data = filter_data(data)

                if foot == 'right':
                    data = flip_matrix(data)

                output_path = output_path_prefix + participant + '/' + trial + '/' + foot + '/'

                cut_data = cut_frame(data, calibrate=False, min2=min2, max2=max2, min1=min1, max1=max1)

                data = all_frames_analysis(cut_data, participant, foot, frames_per_trial[participant][trial], idxs,
                                           output_path=output_path,plot=False)

    compressed_pickle('../processed_data/calibration_constants', calibration_constants)

    compressed_pickle('../processed_data/participant_stimulus_indexes',participant_stimulus_indexes)
