import pandas as pd
from project_insole_constants import *
from project_insole_analysis_information import *
import footsim as fs
from footsim.surface import *
from insole import *
import matplotlib.pyplot as plt
import seaborn as sns


def all_metrics(stomps):
    """ Calculates the percentage mass, CoP location, % body mass change and contact areas

    :param stomps: dictionary containing the time at which the participant stomped in each trial
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
    """

    all_data = {'Force': [], 'Trials': [], 'Foot': [],
                'Participant': [], 'Trial type': [],
                'CoP location': [], 'Contact area percent': [], 'Contact area': [],
                'Change in force': [], 'Trial title': []}
    all_data = pd.DataFrame(all_data)

    trials_no_walking = ['trial05','trial06','trial10','trial11','trial12','trial16','trial17','trial18']


    for foot in feet:

        for trial in trials_no_walking:
            ppts = []
            all_masses = np.array([])
            body_mass_change = np.array([])
            contact_area_percent = np.array([])
            total_contact = np.array([])
            CoP_location = []

            for participant in details[trial]:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    stomp_start = stomps[trial][participant][0] + 5
                    stomp_end = stomps[trial][participant][1] - 5

                    if trial == 'trial05':
                        # calculate percentage mass per frame of data
                        expected_masses = expected_participant_weight(
                            data['Raw data'][:, :, 1000:5500])
                        masses = (expected_masses / participant_mass[participant]) * 100
                        masses[masses < 0] = 0
                        all_masses = np.append(all_masses, masses)

                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, 1000:5500]

                    elif trial == 'trial06':
                        # calculate percentage mass per frame of data
                        expected_masses = expected_participant_weight(
                            data['Raw data'][:, :, 1000:4000])
                        masses = (expected_masses / participant_mass[participant]) * 100
                        masses[masses < 0] = 0
                        all_masses = np.append(all_masses, masses)

                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, 1000:4000]

                    elif trial == 'trial10':
                        stand_end = trial_10_segmentation_times[participant][0]
                        # calculate percentage mass per frame of data
                        expected_masses = expected_participant_weight(
                            data['Raw data'][:, :, stomp_start:stand_end])
                        masses = (expected_masses / participant_mass[participant]) * 100
                        masses[masses < 0] = 0
                        all_masses = np.append(all_masses, masses)

                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, stomp_start:stand_end]

                    elif trial == 'trial11':
                        stand_end = trial_11_segmentation_times[participant][0]

                        # calculate percentage mass per frame of data
                        expected_masses = expected_participant_weight(
                            data['Raw data'][:, :, stomp_start:stand_end])
                        masses = (expected_masses / participant_mass[participant]) * 100
                        masses[masses < 0] = 0
                        all_masses = np.append(all_masses, masses)

                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, stomp_start:stand_end]

                    else:

                        # calculate percentage mass per frame of data
                        expected_masses = expected_participant_weight(data['Raw data'][:,:,stomp_start:stomp_end])
                        masses = (expected_masses / participant_mass[participant]) * 100
                        masses[masses < 0] = 0
                        all_masses = np.append(all_masses, masses)

                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, stomp_start:stomp_end]

                    change = np.zeros((D.shape[0], D.shape[1], D.shape[2] - 1))
                    for p in range(change.shape[2]):
                        change[:, :, p] = D[:, :, p + 1] - D[:, :, p]

                    # calculate the total pressure during the step
                    change_body_mass = expected_participant_weight(change)
                    change_body_mass = (change_body_mass / participant_mass[participant]) * 100
                    change_body_mass = np.append(np.zeros(5), change_body_mass)
                    change_body_mass = moving_average(change_body_mass, 5)
                    # zero = np.zeros(1)
                    # change_body_mass = np.append(zero, change_body_mass)

                    # calculate moving average of change over window size 5 (50Hz)
                    body_mass_change = np.append(body_mass_change, change_body_mass)

                    if trial == 'trial05':
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, 1000:5500]
                            , data['Regions'], data['Reshaped data'][:, 1000:5500])

                    elif trial == 'trial06':
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, 1000:4000]
                            , data['Regions'],
                            data['Reshaped data'][:, 1000:4000])

                    elif trial == 'trial10':
                        stand_end = trial_10_segmentation_times[participant][0]
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, stomp_start:stand_end]
                            , data['Regions'],
                            data['Reshaped data'][:, stomp_start:stand_end])

                    elif trial == 'trial11':
                        stand_end = trial_11_segmentation_times[participant][0]
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, stomp_start:stand_end]
                            , data['Regions'],
                            data['Reshaped data'][:, stomp_start:stand_end])

                    else:

                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, stomp_start:stomp_end]
                            , data['Regions'],
                            data['Reshaped data'][:, stomp_start:stomp_end])

                    summed_contact = total_contact_area(contact_areas)
                    total_contact = np.append(total_contact, summed_contact)
                    total_contact_percentage = total_contact_area(contact_area_as_percent)
                    contact_area_percent = np.append(contact_area_percent, total_contact_percentage)

                    print(D.shape[2])
                    # calculate centre of pressure location
                    loc, y, x = centre_of_pressure(D, threshold=50)
                    CoP_location = localise_CoP(loc, CoP_location)

                    ppts.extend([participant] * len(masses))

                mass = pd.Series(all_masses)

            print(mass.shape)
            print(len(all_masses))
            print(len(ppts))
            print(len(CoP_location))
            print(contact_area_percent.shape)
            print(total_contact.shape)

            indiv = {'Force': mass, 'Trials': [trial] * len(all_masses), 'Foot': [foot] * len(all_masses),
                     'Participant': ppts, 'Trial type': [activity_trial_classification[trial]] * len(all_masses),
                     'CoP location': CoP_location, 'Contact area percent': contact_area_percent, 'Contact area': total_contact,
                     'Change in force': body_mass_change, 'Trial title':
                         [trial_titles[trial]] * len(all_masses)}
            indiv = pd.DataFrame(indiv)

            all_data = all_data.append(indiv)

    return all_data

def regions_all_metrics(idxs, stomps):
    """ Calculates the percentage mass, CoP location, % body mass change and contact areas

    :param idxs: indexes relating to which sensors are active per participant
    :param stomps: dictionary containing the time at which the participant stomped in each trial
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
                regions: one of the 4 coarse regions of the foot
                region_pressures: force experienced by that region of the foot
                reigon change: change in pressure experienced in that region of the foot
    """

    all_data = {'Force': [], 'Trials': [], 'Foot': [],
                'Participant': [], 'Trial type': [],
                'CoP location': [], 'Contact area percent': [], 'Contact area': [],
                'Change in force': [], 'Trial title': [], 'Timepoints': [], 'Regions': [],
                'Regional force': [],
                'Regional change in force': []}
    all_data = pd.DataFrame(all_data)

    trials_no_walking = ['trial05', 'trial06', 'trial10', 'trial11', 'trial12', 'trial16', 'trial17', 'trial18']

    for foot in feet:

        for trial in trials_no_walking:

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

                if calibration_type[participant] == 'point':
                    continue
                else:

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    stomp_start = stomps[trial][participant][0] + 5
                    stomp_end = stomps[trial][participant][1] - 5

                    if trial == 'trial05':
                        D = data['Raw data'][:, :, 1000:5500]
                    elif trial == 'trial06':
                        D = data['Raw data'][:, :, 1000:4000]

                    elif trial == 'trial10':
                        stand_end = trial_10_segmentation_times[participant][0]
                        D = data['Raw data'][:, :, stomp_start:stand_end]

                    elif trial == 'trial11':
                        stand_end = trial_11_segmentation_times[participant][0]
                        D = data['Raw data'][:, :, stomp_start:stand_end]
                    else:
                        D = data['Raw data'][:, :, stomp_start:stomp_end]

                    for r in range(4):
                        timepoints.extend(list(range(0, D.shape[2])))

                    # calculate percentage mass per frame of data
                    expected_masses = expected_participant_weight(D)
                    masses = (expected_masses / participant_mass[participant]) * 100
                    masses[masses < 0] = 0

                    for r in range(4):
                        all_masses = np.append(all_masses, masses)

                    change = np.zeros((D.shape[0], D.shape[1], D.shape[2] - 1))
                    for p in range(change.shape[2]):
                        change[:, :, p] = D[:, :, p + 1] - D[:, :, p]
                    # calculate rate of change per second
                    # change = change / .01
                    # calculate the total pressure during the step
                    change_body_mass = expected_participant_weight(change)
                    change_body_mass = (change_body_mass / participant_mass[participant]) * 100
                    change_body_mass = np.append(np.zeros(5), change_body_mass)
                    change_body_mass = moving_average(change_body_mass, 5)

                    # calculate moving average of window size 5 (50Hz sampling rate)
                    for r in range(4):
                        body_mass_change = np.append(body_mass_change, change_body_mass)

                    reshaped = reshape_data(D, idxs[foot][participant])

                    # calculate contact area of the entire foot
                    contact_areas, contact_area_as_percent = contact_area_per_region(D
                                                                                     , data['Regions'],
                                                                                     reshaped)
                    summed_contact = total_contact_area(contact_areas)
                    total_contact_percentage = total_contact_area(contact_area_as_percent)
                    for r in range(4):
                        contact_area_percent = np.append(contact_area_percent, total_contact_percentage)
                        total_contact = np.append(total_contact, summed_contact)

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

                    # change_body_mass = np.append(np.zeros(5), change_body_mass)
                    change_body_mass = moving_average(change_body_mass, 5)

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

                    ppts.extend([participant] * (D.shape[2] * 4))

                mass = pd.Series(all_masses)
                region_change = pd.Series(regional_change)
                regional_pressure = pd.Series(region_pressures)

            indiv = {'Force': mass, 'Trials': [trial] * len(all_masses),
                     'Foot': [foot] * len(all_masses),
                     'Participant': ppts, 'Trial type': [activity_trial_classification[trial]] * len(all_masses),
                     'CoP location': CoP_location, 'Contact area percent': contact_area_percent, 'Contact area': total_contact,
                     'Change in force': body_mass_change, 'Trial title':
                         [trial_titles[trial]] * len(all_masses),
                     'Timepoints': timepoints,
                     'Regions': region_list, 'Regional force': regional_pressure, 'Regional change in force': region_change}
            indiv = pd.DataFrame(indiv)

            all_data = all_data.append(indiv)

    compressed_pickle('../processed_data/regional_all_metrics', all_data)

    return all_data


def cop_df_creation(stomps):

    CoP_df = {'Trial': [], 'Trial title': [],
              'Participant': [], 'Foot': [], \
              'X_pressure': [], 'Y_pressure': [],
              'X_contact': [], 'Y_contact': [],
              'CoP location': [], 'CoC location': [], 'Timepoints': [], 'CoP inside contact': [], 'Percentile pressure': []}
    CoP_df = pd.DataFrame(CoP_df)

    trials_no_walking = ['trial05', 'trial06', 'trial10', 'trial11', 'trial12', 'trial16', 'trial17', 'trial18']

    for foot in feet:

        for trial in trials_no_walking:

            X_contact = np.array([])
            Y_contact = np.array([])
            X_pressure = np.array([])
            Y_pressure = np.array([])
            participant_list = []
            timepoints = []
            CoP_location = []
            CoC_location = []
            CoP_inside_contact = []
            percetile_pressure = np.array([])

            for participant in details[trial]:
            #for participant in participants:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    stomp_start = stomps[trial][participant][0] + 5
                    stomp_end = stomps[trial][participant][1] - 5

                    if trial == 'trial05':
                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, 1000:5500]
                        reshaped_data = data['Reshaped data'][:, 1000:5500]

                    elif trial == 'trial06':
                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, 1000:4000]
                        reshaped_data = data['Reshaped data'][:, 1000:4000]

                    elif trial == 'trial10':
                        stand_end = trial_10_segmentation_times[participant][0]
                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, stomp_start:stand_end]
                        reshaped_data = data['Reshaped data'][:, stomp_start:stand_end]

                    elif trial == 'trial11':
                        stand_end = trial_11_segmentation_times[participant][0]
                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, stomp_start:stand_end]
                        reshaped_data = data['Reshaped data'][:, stomp_start:stand_end]

                    else:
                        # calculate percentage body mass change per frame of data
                        D = data['Raw data'][:, :, stomp_start:stomp_end]
                        reshaped_data = data['Reshaped data'][:, stomp_start:stomp_end]

                    # calculate centre of pressure location
                    loc_pressure, y, x = centre_of_pressure(D, threshold=50)
                    loc_contact, y, x = centre_of_contact(D, threshold=50)
                    CoP_location = localise_CoP(loc_pressure, CoP_location)
                    CoC_location = localise_CoP(loc_contact, CoC_location)

                    timepoints.extend(list(range(len(loc_pressure))))

                    X_pressure = np.append(X_pressure, loc_pressure[:, 0])
                    Y_pressure = np.append(Y_pressure, loc_pressure[:, 1])
                    X_contact = np.append(X_contact, loc_contact[:, 0])
                    Y_contact = np.append(Y_contact, loc_contact[:, 1])

                    s = data['Stimulus']
                    in_out, percentile = CoP_inside_outside_contact_area(reshaped_data, s, loc_pressure, overall_threshold=50, sensor_threshold=2)
                    CoP_inside_contact = np.append(CoP_inside_contact, in_out)
                    percetile_pressure = np.append(percetile_pressure, percentile)

                    participant_list.extend([participant] * len(loc_pressure))

            for_df = {'Trial': [trial] * len(X_pressure), 'Trial title': [trial_titles[trial]] * len(X_pressure),
                      'Trial type': [activity_trial_classification[trial]] * len(X_pressure),
                      'Participant': participant_list, 'Foot': [foot] * len(X_pressure), \
                      'X_pressure': X_pressure, 'Y_pressure': Y_pressure,
                      'X_contact': X_contact, 'Y_contact': Y_contact,
                      'CoP location': CoP_location, 'CoC location': CoC_location, 'Timepoints': timepoints, 'CoP inside contact': CoP_inside_contact,
                      'Percentile pressure': percetile_pressure}
            df = pd.DataFrame(for_df)

            CoP_df = CoP_df.append(df)
    CoP_df.to_csv('../processed_data/all_CoP_coordinates_df.csv')


def time_each_foot_in_contact_with_ground(filepath_prefix):

    trials = trial_ids[5:-1]

    zero_one_two = {'Data' : [], 'Condition': [], 'Task' : [], 'Task type': []}
    data = pd.DataFrame(zero_one_two)

    conditions = ['no feet','one foot','two feet']

    for trial in trials:

        trial_data = np.array([])
        condition = []

        for participant in details[trial]:

            if calibration_type[participant] == 'point':
                continue
            else:

                # set a different threshold for PPT_014, who's data was noisier
                if participant == 'PPT_014':
                    threshold = 3000
                elif participant == 'PPT_016':
                    threshold = 5000
                else:
                    threshold = 500

                file = filepath_prefix + participant + '/' + trial + '/left/left data.pbz2'
                left_data = decompress_pickle(file)
                left_on_off = foot_on_foot_off(left_data['Total pressure'], threshold=threshold)

                file = filepath_prefix + participant + '/' + trial + '/right/right data.pbz2'
                right_data = decompress_pickle(file)
                right_on_off = foot_on_foot_off(right_data['Total pressure'], threshold=threshold)

                combined = left_on_off + right_on_off

                zero_on = len(np.where(combined == 0)[0]) / len(combined)
                one_on = len(np.where(combined == 1)[0]) / len(combined)
                two_on = len(np.where(combined == 2)[0]) / len(combined)


                trial_data = np.append(trial_data, np.array([zero_on, one_on, two_on]))
                condition.extend(conditions)

        trial_df = {'Data': trial_data, 'Condition': condition, 'Task': [trial_titles[trial]] * len(trial_data), 'Task type' : [activity_trial_classification[trial]] * len(trial_data)}
        trial_df = pd.DataFrame(trial_df)

        data = data.append(trial_df)


    sns.set_palette(foot_on_ground_color_palette)
    plt.figure(figsize=(12,15), dpi=600)
    sns.barplot(x=data['Task'], y=data['Data']*100, hue=data['Condition'], ci='sd', errwidth=.75, saturation=.75, alpha=0.75)
    plt.ylim(0,110)
    plt.xticks(rotation=80)
    plt.savefig('../individual_figures/number_of_feet_on_the_ground.png')

    sns.set_palette(foot_on_ground_color_palette)
    plt.figure(figsize=(12, 15), dpi=600)
    sns.barplot(x=data['Task type'], y=data['Data'] * 100, hue=data['Condition'], ci='sd', errwidth=.75, saturation=.75,
                alpha=0.75)
    plt.ylim(0, 110)
    plt.savefig('../individual_figures/number_of_feet_on_the_ground_per_category.png')

    return data
