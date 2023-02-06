import pandas as pd
import numpy as np
from insole import *
from project_insole_constants import *
from project_insole_analysis_information import *
import footsim as fs
from footsim.surface import *


def find_trial_14_turn_indexes(participant):
    """ Find and return the time during which the paricipant was
    turning in trial 14

    :param participant: participant id

    :return:
        turn_idxs: time points relating to when the turn started and ended
    """

    turn_idxs = list(range(trial_14_segmentation_times[participant][0], trial_14_segmentation_times[participant][1]))

    return turn_idxs

def normalize_step_length_across_participants_with_exclusions(filepath_prefix, stomps):
    """ Normalises each step from each participant to the same length

    :param longest: longest
    :param calibration_constants:
    :param filepath_prefix:
    :return:
    """
    trials = ['trial07', 'trial08', 'trial09', 'trial13', 'trial14', 'trial15', 'trial19']

    normalized_steps = {}
    j = 0

    # loop through walking trials specified
    for trial in trials:
        normalized_steps[trial] = {}

        # loop through the participants in each trial
        for participant in details[trial]:

            normalized_steps[trial][participant] = {}

            i = 0

            # loop through each foot
            for foot in feet:

                normalized_steps[trial][participant][foot] = {}

                # exclude participants that used point calibration
                if calibration_type[participant] == 'point':
                    continue
                else:

                    # load data for the participant
                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    # set a different threshold for PPT_014, who's data was noisier
                    if participant == 'PPT_014':
                        threshold = 3000
                    elif participant == 'PPT_016':
                        threshold = 5000
                    else:
                        threshold = 500

                    D = data['Raw data']
                    D[D < 1.] = 0.0
                    # check for steps in the data
                    under_threshold, step_start, index_differences, total_pressure = check4steps(
                        D, threshold=threshold, plot=False)

                    # set the filepath for IMU segmentation
                    segmentaion_file_path = '../raw_data/IMU/Raw data/' + participant + '/' + trial + '/Segmentation times.csv'

                    # if participant is PPT_004, set a different file path (as they stomped the right foot)
                    if participant != 'PPT_004':
                        gryo_data_filename = '../raw_data/IMU/Raw data/' + participant + '/' + trial + '/left_foot_accel.csv'
                    else:
                        gryo_data_filename = '../raw_data/IMU/Raw data/' + participant + '/' + trial + '/right_foot_accel.csv'

                    # find the time points in the trial during which the participant was turning
                    # if the trial is trial14 (stairs), use a different, manual segmentation
                    if trial != 'trial14':
                        turn_idxs = find_turning_idxs(segmentaion_file_path, gryo_data_filename)
                        segment_pressure_data(data['Total pressure'], segmentaion_file_path, gryo_data_filename, stomps[trial][participant][0])
                        stomp = stomps[trial][participant][0]

                        turn_idxs = (turn_idxs) + stomp
                    else:
                        turn_idxs = find_trial_14_turn_indexes(participant)


                    # extract each step from the data
                    all_steps, total_step_frame = average_after_exclusions(D, \
                                                                           foot, step_start, under_threshold, index_differences, turn_idxs)

                    # normalizes the step length to 100 time points
                    all_steps_new, total_step_frame_new = normalize_step_length_given_longest(all_steps,
                                                                                              total_step_frame)


                    normalized_steps[trial][participant][foot]['total step pressure'] = all_steps_new
                    normalized_steps[trial][participant][foot]['total step frame'] = total_step_frame_new

                i += 1

        j += 1

    return normalized_steps

def all_metrics_steps(normalized_steps, idxs):
    """ Calculates the percentage mass, CoP location, % body mass change and contact areas

    :param normalized_steps: dictionary containing steps normalized to 100 time points
    :param idxs: indexes relating to which sensors are active per participant
    :return:
        all_data (pd.DataFrame):
            columns:
                percent body mass: force experienced
                trials: trial id
                foot: left or right
                CoP location: T, M, A, H - coarse reigon that the CoP is located in
                Contact area percent: area of foot in contact with the ground
                change in body mass: change in force over a 0.1s time window
                trial title: name of the trial
                timepoints: 0-100
    """

    trials = ['trial07','trial08','trial09','trial13','trial14','trial15','trial19']

    all_data = {'Force': [], 'Trials': [], 'Foot': [],
                'Participant': [], 'Trial type': [],
                'CoP location': [], 'Contact area percent': [], 'Contact area': [],
                'Change in force': [], 'Trial title': [], 'Timepoints':[], 'Step number': []}
    all_data = pd.DataFrame(all_data)

    for foot in feet:

        for trial in normalized_steps:
            ppts = []
            all_masses = np.array([])
            body_mass_change = np.array([])
            contact_area_percent = np.array([])
            total_contact = np.array([])
            CoP_location = []
            timepoints = []
            step_numbers = []

            for participant in details[trial]:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    # remove steps that are now zero
                    remove = []
                    for q in range(normalized_steps[trial][participant][foot]['total step frame'].shape[0]):
                        if np.sum(normalized_steps[trial][participant][foot]['total step frame'][q, :, :, :]) == 0.0:
                            remove.append(q)
                    total_step_frame = np.delete(normalized_steps[trial][participant][foot]['total step frame'], remove,
                                                 axis=0)

                    # loop through each of the normalized steps
                    for s in range(total_step_frame.shape[0]):
                        # set D to be the raw data for the step
                        D = total_step_frame[s]
                        # re-organise the matrix so that the time dimension is axis 2
                        D = np.moveaxis(D, 0, 2)

                        timepoints.extend(list(range(0, 100)))
                        step_numbers.extend([s] * 100)

                        # calculate percentage mass per frame of data
                        expected_masses = expected_participant_weight(D)
                        masses = (expected_masses / participant_mass[participant]) * 100
                        masses[masses < 0] = 0
                        all_masses = np.append(all_masses, masses)

                        change = np.zeros((D.shape[0], D.shape[1], D.shape[2] - 1))
                        for p in range(change.shape[2]):
                            change[:, :, p] = D[:, :, p + 1] - D[:, :, p]

                        # calculate the total pressure during the step
                        change_body_mass = expected_participant_weight(change)
                        change_body_mass = (change_body_mass / participant_mass[participant]) * 100
                        change_body_mass = np.append(np.zeros(5), change_body_mass)
                        change_body_mass = moving_average(change_body_mass, 5)

                        body_mass_change = np.append(body_mass_change, change_body_mass)

                        reshaped = reshape_data(D, idxs[foot][participant])

                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(D
                            , data['Regions'], reshaped)

                        summed_contact = total_contact_area(contact_areas)
                        total_contact = np.append(total_contact, summed_contact)
                        total_contact_percentage = total_contact_area(contact_area_as_percent)
                        contact_area_percent = np.append(contact_area_percent, total_contact_percentage)

                        # calculate centre of pressure location
                        loc, y, x = centre_of_pressure(D, threshold=50)
                        CoP_location = localise_CoP(loc, CoP_location)

                        ppts.extend([participant] * 100)

                mass = pd.Series(all_masses)

            indiv = {'Force': mass, 'Trials': [trial] * len(all_masses), 'Foot': [foot] * len(all_masses),
                     'Participant': ppts, 'Trial type': [activity_trial_classification[trial]] * len(all_masses),
                     'CoP location': CoP_location, 'Contact area percent': contact_area_percent, 'Contact area': total_contact,
                     'Change in force': body_mass_change, 'Trial title':
                         [trial_titles[trial]] * len(all_masses), 'Timepoints':timepoints, 'Step number': step_numbers}
            indiv = pd.DataFrame(indiv)

            all_data = all_data.append(indiv)

    return all_data

def regions_all_metrics_steps(normalized_steps, idxs):
    """ Calculates the percentage mass, CoP location, % body mass change and contact areas

    :param normalized_steps: dictionary containing steps normalized to 100 time points
    :param idxs: indexes relateing to which sensors are active per participant
    :return:
        all_data (pd.DataFrame):
            columns:
                percent body mass: force experienced
                trials: trial id
                foot: left or right
                CoP location: T, M, A, H - coarse reigon that the CoP is located in
                Contact area percent: area of foot in contact with the ground
                change in body mass: change in force over a 0.1s time window
                trial title: name of the trial
                timepoints: 0-100
                regions: one of the 4 coarse regions of the foot
                region_pressures: force experienced by that region of the foot
                reigon change: change in pressure experienced in that region of the foot
    """

    all_data = {'Force': [], 'Trials': [], 'Foot': [],
                'Participant': [], 'Trial type': [],
                'CoP location': [], 'Contact area percent': [], 'Contact area': [],
                'Change in force': [], 'Trial title': [], 'Timepoints':[], 'Regions':[], 'Regional force': [],
                'Regional change in force': []}
    all_data = pd.DataFrame(all_data)

    for foot in feet:
        print(foot)

        for trial in normalized_steps:
            print(trial)
            ppts = []
            all_masses = np.array([])
            body_mass_change = np.array([])
            contact_area_percent = np.array([])
            total_contact = np.array([])
            CoP_location = []
            timepoints = []
            region_list = []
            region_pressures = np.array([])
            regional_change = np.array([])

            for participant in details[trial]:
                print(participant)

                if calibration_type[participant] == 'point':
                    continue
                else:
                    print(participant)

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    # remove steps that are now zero
                    remove = []
                    for q in range(normalized_steps[trial][participant][foot]['total step frame'].shape[0]):
                        if np.sum(normalized_steps[trial][participant][foot]['total step frame'][q, :, :, :]) == 0.0:
                            remove.append(q)
                    total_step_frame = np.delete(normalized_steps[trial][participant][foot]['total step frame'], remove,
                                                 axis=0)

                    # loop through each of the normalized steps
                    for s in range(total_step_frame.shape[0]):
                        # set D to be the raw data for the step
                        D = total_step_frame[s]
                        # re-organise the matrix so that the time dimension is axis 2
                        D = np.moveaxis(D, 0, 2)

                        for r in range(4):
                            timepoints.extend(list(range(0, 100)))


                        # calculate percentage mass per frame of data
                        expected_masses = expected_participant_weight(D)
                        masses = (expected_masses / participant_mass[participant]) * 100
                        masses[masses < 0] = 0

                        for r in range(4):
                            all_masses = np.append(all_masses, masses)



                        change = np.zeros((D.shape[0], D.shape[1], D.shape[2] - 1))
                        for p in range(change.shape[2]):
                            change[:, :, p] = D[:, :, p + 1] - D[:, :, p]

                        # calculate the total pressure during the step
                        change_body_mass = expected_participant_weight(change)
                        change_body_mass = (change_body_mass / participant_mass[participant]) * 100
                        change_body_mass = np.append(np.zeros(5), change_body_mass)
                        change_body_mass = moving_average(change_body_mass, 5)

                        for r in range(4):
                            body_mass_change = np.append(body_mass_change, change_body_mass)

                        reshaped = reshape_data(D, idxs[foot][participant])

                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(D
                            , data['Regions'], reshaped)

                        summed_contact = total_contact_area(contact_areas)
                        total_contact_percentage = total_contact_area(contact_area_as_percent)

                        for r in range(4):
                            total_contact = np.append(total_contact, summed_contact)
                            contact_area_percent = np.append(contact_area_percent, total_contact_percentage)

                            # calculate centre of pressure location
                            loc, y, x = centre_of_pressure(D, threshold=50)
                            CoP_location = localise_CoP(loc, CoP_location)

                                        ### REGIONAL BREAKDOWN

                        # reshape the matrix to remove sensors off of the foot
                        reshaped = reshape_data(D, idxs[foot][participant])

                        # calculate the average and total  pressure per region & contact percentage per region
                        av_pressure_per_region, contact_percentage_per_region, total_pressure_per_region = \
                            average_pressure_per_region(reshaped, data['Regions'], foot, participant)

                        # calculate the expected participant mass per region
                        expected_mass_per_region = expected_participant_mass_per_region(
                            total_pressure_per_region,
                            data['Regions'], reshaped)

                        toes, metatarsal, arch, heel = expected_mass_coarse_regions(
                            participant_mass[participant], expected_mass_per_region, remove=False)

                        # append each of the general regions into one array
                        region_pressures = np.append(region_pressures, toes)
                        region_pressures = np.append(region_pressures, metatarsal)
                        region_pressures = np.append(region_pressures, arch)
                        region_pressures = np.append(region_pressures, heel)


                        # CHANGE AT THE REGIONAL LEVEL

                        # reshape the matrix to remove sensors off of the foot
                        reshaped_change = reshape_data(change, idxs[foot][participant])

                        # calculate the average and total  pressure per region & contact percentage per region
                        av_pressure_per_region_change, contact_percentage_per_region_change, total_pressure_per_region_change = \
                            average_pressure_per_region(reshaped_change, data['Regions'], foot, participant)

                        # calculate the expected participant mass per region
                        expected_mass_per_region_change = expected_participant_mass_per_region(
                            total_pressure_per_region_change,
                            data['Regions'], reshaped_change)

                        toes_change, metatarsal_change, arch_change, heel_change = expected_mass_coarse_regions(
                            participant_mass[participant], expected_mass_per_region_change, remove=False)

                        toes_change = np.append(np.zeros(5), toes_change)
                        metatarsal_change = np.append(np.zeros(5), metatarsal_change)
                        arch_change = np.append(np.zeros(5), arch_change)
                        heel_change = np.append(np.zeros(5), heel_change)

                        # change at toes  # calculate moving average of window size 5 (50Hz sampling rate)
                        toes_change = moving_average(toes_change, 5)
                        regional_change = np.append(regional_change, toes_change)
                        # change at metatarsals  # calculate moving average of window size 5 (50Hz sampling rate)
                        metatarsal_change = moving_average(metatarsal_change, 5)
                        regional_change = np.append(regional_change, metatarsal_change)
                        # change at arch  # calculate moving average of window size 5 (50Hz sampling rate)
                        arch_change = moving_average(arch_change, 5)
                        regional_change = np.append(regional_change, arch_change)
                        # change at heel  # calculate moving average of window size 5 (50Hz sampling rate)
                        heel_change = moving_average(heel_change, 5)
                        regional_change = np.append(regional_change, heel_change)

                        # append the region name to a list
                        region_list.extend(['toes'] * len(toes))
                        region_list.extend(['metatarsal'] * len(metatarsal))
                        region_list.extend(['arch'] * len(arch))
                        region_list.extend(['heel'] * len(heel))

                        ppts.extend([participant] * 400)

                mass = pd.Series(all_masses)
                region_change = pd.Series(regional_change)
                regional_pressure = pd.Series(region_pressures)

            indiv = {'Force': mass, 'Trials': [trial] * len(all_masses), 'Foot': [foot] * len(all_masses),
                     'Participant': ppts, 'Trial type': [activity_trial_classification[trial]] * len(all_masses),
                     'CoP location': CoP_location, 'Contact area percent': contact_area_percent, 'Contact area': total_contact,
                     'Change in force': body_mass_change, 'Trial title':
                         [trial_titles[trial]] * len(all_masses), 'Timepoints': timepoints,
                     'Regions': region_list, 'Regional force': regional_pressure, 'Regional change in force': region_change}
            indiv = pd.DataFrame(indiv)

            all_data = all_data.append(indiv)

    compressed_pickle('../processed_data/regional_all_metrics_steps', all_data)

    return all_data

def walking_cop_df_creation(normalized_steps, idxs):
    CoP_df = {'Trial': [], 'Trial title': [],
              'Participant': [], 'Foot': [], \
              'X_pressure': [], 'Y_pressure': [],
              'X_contact': [], 'Y_contact': [],
              'CoP location': [], 'CoC location': [], 'Timepoints': [], 'CoP inside contact': []}
    CoP_df = pd.DataFrame(CoP_df)

    selected_trials = ['trial07', 'trial08', 'trial09', 'trial13', 'trial14', 'trial15', 'trial19']

    for foot in feet:

        for trial in normalized_steps:

            X_contact = np.array([])
            Y_contact = np.array([])
            X_pressure = np.array([])
            Y_pressure = np.array([])
            participant_list = []
            timepoints = []
            CoP_location = []
            CoC_location = []
            CoP_inside_contact = []
            percentile_pressure = np.array([])

            for participant in details[trial]:

                file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                data = decompress_pickle(file)

                ppt_idxs = idxs[foot][participant]

                if calibration_type[participant] == 'point':
                    continue
                else:

                    # remove steps that are now zero
                    remove = []
                    for q in range(normalized_steps[trial][participant][foot]['total step frame'].shape[0]):
                        if np.sum(normalized_steps[trial][participant][foot]['total step frame'][q, :, :, :]) == 0.0:
                            remove.append(q)
                    total_step_frame = np.delete(normalized_steps[trial][participant][foot]['total step frame'], remove,
                                                 axis=0)

                    # loop through the steps
                    for s in range(total_step_frame.shape[0]):
                        D = total_step_frame[s]
                        D = np.moveaxis(D, 0, 2)

                        # calculate centre of pressure location
                        loc_pressure, y, x = centre_of_pressure(D, threshold=50)
                        loc_contact, y, x = centre_of_contact(D, threshold=50)
                        CoP_location = localise_CoP(loc_pressure, CoP_location)
                        CoC_location = localise_CoP(loc_contact, CoC_location)

                        timepoints.extend(list(range(100)))

                        X_pressure = np.append(X_pressure, loc_pressure[:, 0])
                        Y_pressure = np.append(Y_pressure, loc_pressure[:, 1])
                        X_contact = np.append(X_contact, loc_contact[:, 0])
                        Y_contact = np.append(Y_contact, loc_contact[:, 1])

                        reshaped_data = reshape_data(D, ppt_idxs)
                        s = data['Stimulus']
                        in_out, percentile = CoP_inside_outside_contact_area(reshaped_data, s, loc_pressure, overall_threshold=50, sensor_threshold=2)
                        CoP_inside_contact = np.append(CoP_inside_contact, in_out)
                        percentile_pressure = np.append(percentile_pressure, percentile)

                        participant_list.extend([participant] * 100)


            for_df = {'Trial': [trial] * len(X_pressure), 'Trial title': [trial_titles[trial]] * len(X_pressure),
                      'Trial type': [activity_trial_classification[trial]] * len(X_pressure),
                      'Participant': participant_list, 'Foot': [foot] * len(X_pressure),
                      'X_pressure': X_pressure, 'Y_pressure': Y_pressure,
                      'X_contact': X_contact, 'Y_contact': Y_contact,
                      'CoP location': CoP_location, 'CoC location': CoC_location, 'Timepoints': timepoints, 'CoP inside contact': CoP_inside_contact,
                      'Percentile pressure': percentile_pressure}

            df = pd.DataFrame(for_df)

            CoP_df = CoP_df.append(df)
    CoP_df.to_csv('../processed_data/walking_CoP_coordinates_df.csv')