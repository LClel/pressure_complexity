from insole import *
import numpy as np
import pandas as pd
from project_insole_constants import *
from project_insole_analysis_information import *
import matplotlib.pyplot as plt
from figures_functions import *
import footsim as fs

def project_insole_calibration(separation_indexes, calibration_constants):
    """Run and print the calibration tests. Calculates the percentage of mass captured by the insoles
    during the 3 calibration trials:
        trial01: both feet, left foot, right foot, both feet
        trial02: flat, tiptoe, heel, flat
        trial20: repetition of trial01, but at the end of the experiment

    Args:
        separation_indexes: timepoints during which stance chanegd - in project_insole_information
        calibration_constants: file containing values used to recalibrate data

    Returns:

    """


    filepath_prefix = '../preprocessed_data/'
    before_after = ['pre','post']

    both_trials = {}

    # loop through the 3 calibration trials
    for trial in separation_indexes:

        # create empty array to store the data for the trial
        data = np.array([])

        pre_post = []
        trial_1_20 = []
        condition = []
        ppts = []

        if trial != 'trial02':

            # loop through each participant in the trials
            for participant in separation_indexes[trial]:

                # get the participant mass
                actual_mass = [participant_mass[participant]]

                # exclude the participant if point calibration was run
                if calibration_type[participant] == 'point':
                    continue
                else:

                    # load in the data for the left foot
                    foot = 'left'
                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    left_data = decompress_pickle(file)['Raw data']
                    left_data_pre = left_data / calibration_constants['left'][participant]

                    # load in the data for the right foot
                    foot = 'right'
                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    right_data = decompress_pickle(file)['Raw data']
                    right_data_pre = right_data / calibration_constants['right'][participant]

                    post_left_only_mass, post_right_only_mass, post_both_mass = \
                        left_right_both_mass(left_data, right_data, separation_indexes, actual_mass, participant, trial)

                    pre_left_only_mass, pre_right_only_mass, pre_both_mass = \
                        left_right_both_mass(left_data_pre, right_data_pre, separation_indexes, actual_mass, participant, trial)

                    # loop through before and after the recalibration
                    for case in before_after:
                        if case == 'pre':
                            trial_1_20.extend([trial] * 500)
                            pre_post.extend([case] * 1500)

                            data = np.append(data, pre_left_only_mass)
                            condition.extend(['left'] * 500)
                            data = np.append(data, pre_both_mass)
                            condition.extend(['both'] * 500)
                            data = np.append(data, pre_right_only_mass)
                            condition.extend(['right'] * 500)

                            ppts.extend([participant] * 1500)

                        elif case == 'post':
                            trial_1_20.extend([trial] * 500)
                            pre_post.extend([case] * 1500)

                            data = np.append(data, post_left_only_mass)
                            condition.extend(['left'] * 500)
                            data = np.append(data, post_both_mass)
                            condition.extend(['both'] * 500)
                            data = np.append(data, post_right_only_mass)
                            condition.extend(['right'] * 500)

                            ppts.extend([participant] * 1500)

                        else:
                            continue

            for_df = {'Calibration': pre_post, 'Condition': condition, 'Data': data, 'Participant': ppts}
            df = pd.DataFrame(for_df)
            both_trials[trial] = df


        # for trial02
        elif trial == 'trial02':

            # loop through the participants in the trial
            for participant in separation_indexes[trial]:

                # get mass of the participant
                actual_mass = [participant_mass[participant]]

                # exclude the participant if point calibration was run
                if calibration_type[participant] == 'point':
                    continue
                else:

                    # load the data for the left foot
                    foot = 'left'
                    file = filepath_prefix + participant + '/trial02/' + foot + '/' + foot + ' data.pbz2'
                    left_data = decompress_pickle(file)['Raw data']
                    left_data_pre = left_data / calibration_constants['left'][participant]

                    # load the data for the right foot
                    foot = 'right'
                    file = filepath_prefix + participant + '/trial02/' + foot + '/' + foot + ' data.pbz2'
                    right_data = decompress_pickle(file)['Raw data']
                    right_data_pre = right_data / calibration_constants['right'][participant]

                    post_heel_mass, post_tiptoe_mass = \
                        heel_tiptoe_mass(left_data, right_data, separation_indexes, actual_mass, participant, trial)

                    pre_heel_mass, pre_tiptoe_mass = \
                        heel_tiptoe_mass(left_data_pre, right_data_pre, separation_indexes, actual_mass, participant, trial)

                    # loop through before and after the recalibration
                    for case in before_after:
                        if case == 'pre':
                            trial_1_20.extend([trial] * 500)
                            pre_post.extend([case] * 1000)

                            data = np.append(data, pre_tiptoe_mass)
                            condition.extend(['tiptoe'] * 500)
                            data = np.append(data, pre_heel_mass)
                            condition.extend(['heel'] * 500)

                            ppts.extend([participant] * 1000)


                        elif case == 'post':
                            trial_1_20.extend([trial] * 500)
                            pre_post.extend([case] * 1000)

                            data = np.append(data, post_tiptoe_mass)
                            condition.extend(['tiptoe'] * 500)
                            data = np.append(data, post_heel_mass)
                            condition.extend(['heel'] * 500)

                            ppts.extend([participant] * 1000)


                        else:
                            continue


        for_df = {'Calibration': pre_post, 'Condition': condition, 'Data': data, 'Participant': ppts}
        df = pd.DataFrame(for_df)
        both_trials[trial] = df

    flierprops = dict(marker='*', markerfacecolor='r', markersize=.2,
                      linestyle='none', markeredgecolor='r')

    plt.close("all")
    colours = ['darkorange','darkorchid']
    sns.set_palette(colours)
    plt.figure(figsize=(13,8))
    plt.subplot(1,3,1)
    plt.suptitle('Percentage of participant mass captured by the insoles')
    plt.title('trial 1')
    sns.boxplot(x=both_trials['trial01']['Condition'], y=both_trials['trial01']['Data'], \
                hue=both_trials['trial01']['Calibration'], flierprops=flierprops)
    plt.ylabel('percentage of mass')
    plt.ylim(0,200)

    plt.subplot(1,3,2)
    plt.title('trial 02')
    sns.boxplot(x=both_trials['trial02']['Condition'], y=both_trials['trial02']['Data'],
                hue=both_trials['trial02']['Calibration'], flierprops=flierprops)
    plt.ylabel('percentage of mass')
    plt.ylim(0,300)

    plt.subplot(1,3,3)
    plt.title('trial 20')
    sns.boxplot(x=both_trials['trial20']['Condition'], y=both_trials['trial20']['Data'], \
                hue=both_trials['trial20']['Calibration'], flierprops=flierprops)
    plt.ylabel('percentage of mass')
    plt.ylim(0,200)
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('../paper_figures/calibration.png')

    return both_trials

def raw_pressure_all(idxs, stomps):
    """ Calculate raw pressure (kPa) and force (N) for all data

    Args:
        idxs: file containing locations of active sensors for each participant
        stomps: timepoints at which stomp occurred at the start and end of trial

    Returns:

    """

    all_data = {'Total pressure': [], 'Trials': [], 'Foot': [],
                'Participant': [], 'Trial type': [],
                'Total pressure per region': [], 'Average pressure per region': [],'Contact area': [],
                'Trial title': [], 'Force (N)': [], 'Total force per region': [],
                     'Average force per region': []}
    all_data = pd.DataFrame(all_data)

    for foot in feet:

        for trial in trial_ids[5:-1]:
            ppts = []
            all_pressures = np.array([])
            all_forces = np.array([])
            total_contact = np.array([])
            total_region_pressures = np.array([])
            total_region_forces = np.array([])
            average_region_pressures = np.array([])
            average_region_forces = np.array([])
            region_list = []

            for participant in details[trial]:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    stomp_start = stomps[trial][participant][0] + 5
                    stomp_end = stomps[trial][participant][1] - 5

                    if trial == 'trial05':
                        pressures = data['Total pressure'][1000:5500]
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, 1000:5500]
                            , data['Regions'], data['Reshaped data'][:, 1000:5500])
                        D = data['Raw data'][:, :, 1000:5500]


                    elif trial == 'trial06':
                        pressures = data['Total pressure'][1000:4000]
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, 1000:4000]
                            , data['Regions'], data['Reshaped data'][:, 1000:4000])
                        D = data['Raw data'][:, :, 1000:4000]

                    elif trial == 'trial10':
                        stand_end = trial_10_segmentation_times[participant][0]
                        pressures = data['Total pressure'][stomp_start:stand_end]
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, stomp_start:stand_end]
                            , data['Regions'], data['Reshaped data'][:, stomp_start:stand_end])
                        D = data['Raw data'][:, :, stomp_start:stand_end]

                    elif trial == 'trial11':
                        stand_end = trial_11_segmentation_times[participant][0]
                        pressures = data['Total pressure'][stomp_start:stand_end]
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, stomp_start:stand_end]
                            , data['Regions'], data['Reshaped data'][:, stomp_start:stand_end])
                        D = data['Raw data'][:, :, stomp_start:stand_end]

                    else:
                        pressures = data['Total pressure'][stomp_start:stomp_end]
                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(
                            data['Raw data'][:, :, stomp_start:stomp_end]
                            , data['Regions'], data['Reshaped data'][:, stomp_start:stomp_end])
                        D = data['Raw data'][:, :, stomp_start:stomp_end]


                    reshaped = reshape_data(D, idxs[foot][participant])

                    summed_contact = total_contact_area(contact_areas)
                    for r in range(4):
                        total_contact = np.append(total_contact, summed_contact)
                        all_pressures = np.append(all_pressures, pressures)

                    av_pressure_per_region, contact_percentage_per_region, total_pressure_per_region = \
                        average_pressure_per_region(reshaped, data['Regions'], foot, participant)

                    total_toes, total_metatarsal, total_arch, total_heel = total_pressure_coarse_regions(total_pressure_per_region, remove=False)

                    # append each of the general regions into one array
                    total_region_pressures = np.append(total_region_pressures, total_toes)
                    total_region_pressures = np.append(total_region_pressures, total_metatarsal)
                    total_region_pressures = np.append(total_region_pressures, total_arch)
                    total_region_pressures = np.append(total_region_pressures, total_heel)

                    av_toes, av_metatarsal, av_arch, av_heel = average_pressure_coarse_regions(
                        av_pressure_per_region, remove=False)

                    # append each of the general regions into one array
                    average_region_pressures = np.append(average_region_pressures, av_toes)
                    average_region_pressures = np.append(average_region_pressures, av_metatarsal)
                    average_region_pressures = np.append(average_region_pressures, av_arch)
                    average_region_pressures = np.append(average_region_pressures, av_heel)



                    force_matrix = pressure_to_force(D)
                    total_force = np.zeros(D.shape[2])
                    for m in range(D.shape[2]):
                        total_force[m] = np.sum(force_matrix[:,:,m])

                    reshaped_force = reshape_data(force_matrix, idxs[foot][participant])

                    for r in range(4):
                        all_forces = np.append(all_forces, total_force)

                    av_force_per_regin, contact_percentage_per_region, total_force_per_region = \
                        average_pressure_per_region(reshaped_force, data['Regions'], foot, participant)

                    total_force_toes, total_force_metatarsal, total_force_arch, total_force_heel = total_pressure_coarse_regions(
                        total_force_per_region, remove=False)

                    # append each of the general regions into one array
                    total_region_forces = np.append(total_region_forces, total_force_toes)
                    total_region_forces = np.append(total_region_forces, total_force_metatarsal)
                    total_region_forces = np.append(total_region_forces, total_force_arch)
                    total_region_forces = np.append(total_region_forces, total_force_heel)

                    av_force_toes, av_force_metatarsal, av_force_arch, av_force_heel = average_pressure_coarse_regions(
                        av_force_per_regin, remove=False)

                    # append each of the general regions into one array
                    average_region_forces = np.append(average_region_forces, av_force_toes)
                    average_region_forces = np.append(average_region_forces, av_force_metatarsal)
                    average_region_forces = np.append(average_region_forces, av_force_arch)
                    average_region_forces = np.append(average_region_forces, av_force_heel)

                    # append the region name to a list
                    region_list.extend(['toes'] * len(av_toes))
                    region_list.extend(['metatarsal'] * len(av_metatarsal))
                    region_list.extend(['arch'] * len(av_arch))
                    region_list.extend(['heel'] * len(av_heel))

                    ppts.extend([participant] * (D.shape[2] * 4))

                total_pressure = pd.Series(all_pressures)
                total_forces = pd.Series(all_forces)
                regional_pressure = pd.Series(total_region_pressures)
                average_pressure = pd.Series(average_region_pressures)
                regional_force = pd.Series(total_region_forces)
                average_force = pd.Series(average_region_forces)


            indiv = {'Total pressure': total_pressure, 'Trials': [trial] * len(total_pressure), 'Foot': [foot] * len(total_pressure),
                        'Participant': ppts, 'Trial type': [activity_trial_classification[trial]] * len(all_pressures), 'Region': region_list,
                        'Total pressure per region': regional_pressure, 'Average pressure per region': average_pressure, 'Contact area': total_contact,
                        'Trial title': [trial_titles[trial]] * len(all_pressures), 'Force (N)': total_forces, 'Total force per region': regional_force,
                     'Average force per region': average_force}
            indiv = pd.DataFrame(indiv)

            all_data = all_data.append(indiv)

    compressed_pickle('../processed_data/regional_pressure_validation', all_data)

    return all_data

def raw_pressure_steps(normalized_steps, idxs):
    """ Calculate raw pressure (kPa) and force (N) for walking trials using normalized steps data

    Args:
        idxs: file containing locations of active sensors for each participant
        normalized_steps: each step for each participant in each walking task normalized to 100 timepoints

    Returns:

    """

    all_data = {'Total pressure': [], 'Trials': [], 'Foot': [],
                'Participant': [], 'Trial type': [],
                'Total pressure per region': [], 'Average pressure per region': [],'Contact area': [],
                'Trial title': []}
    all_data = pd.DataFrame(all_data)

    for foot in feet:

        for trial in normalized_steps:
            ppts = []
            all_pressures = np.array([])
            total_contact = np.array([])
            total_region_pressures = np.array([])
            average_region_pressures = np.array([])
            region_list = []
            timepoints = []

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
                    total_step_pressure = np.delete(normalized_steps[trial][participant][foot]['total step pressure'], remove,
                                                 axis=0)


                    # loop through each of the normalized steps
                    for s in range(total_step_frame.shape[0]):
                        # set D to be the raw data for the step
                        D = total_step_frame[s]
                        # re-organise the matrix so that the time dimension is axis 2
                        D = np.moveaxis(D, 0, 2)

                        pressures = total_step_pressure[s]

                        timepoints.extend(list(range(0, 100)))

                        reshaped = reshape_data(D, idxs[foot][participant])

                        # calculate contact area of the entire foot
                        contact_areas, contact_area_as_percent = contact_area_per_region(D, data['Regions'], reshaped)
                        summed_contact = total_contact_area(contact_areas)
                        for r in range(4):
                            total_contact = np.append(total_contact, summed_contact)
                            all_pressures = np.append(all_pressures, pressures)


                        av_pressure_per_region, contact_percentage_per_region, total_pressure_per_region = \
                            average_pressure_per_region(reshaped, data['Regions'], foot, participant)

                        total_toes, total_metatarsal, total_arch, total_heel = total_pressure_coarse_regions(total_pressure_per_region, remove=False)

                        # append each of the general regions into one array
                        total_region_pressures = np.append(total_region_pressures, total_toes)
                        total_region_pressures = np.append(total_region_pressures, total_metatarsal)
                        total_region_pressures = np.append(total_region_pressures, total_arch)
                        total_region_pressures = np.append(total_region_pressures, total_heel)

                        av_toes, av_metatarsal, av_arch, av_heel = average_pressure_coarse_regions(
                            av_pressure_per_region, remove=False)

                        # append each of the general regions into one array
                        average_region_pressures = np.append(average_region_pressures, av_toes)
                        average_region_pressures = np.append(average_region_pressures, av_metatarsal)
                        average_region_pressures = np.append(average_region_pressures, av_arch)
                        average_region_pressures = np.append(average_region_pressures, av_heel)

                        # append the region name to a list
                        region_list.extend(['toes'] * len(av_toes))
                        region_list.extend(['metatarsal'] * len(av_metatarsal))
                        region_list.extend(['arch'] * len(av_arch))
                        region_list.extend(['heel'] * len(av_heel))

                        ppts.extend([participant] * (D.shape[2] * 400))

                total_pressure = pd.Series(all_pressures)
                regional_pressure = pd.Series(total_region_pressures)
                average_pressure = pd.Series(average_region_pressures)

            indiv = {'Total pressure': total_pressure, 'Trials': [trial] * len(total_pressure), 'Foot': [foot] * len(total_pressure),
                        'Participant': ppts, 'Trial type': [activity_trial_classification[trial]] * len(all_pressures), 'Region': region_list,
                        'Total pressure per region': regional_pressure, 'Average pressure per region': average_pressure, 'Contact area': total_contact,
                        'Trial title': [trial_titles[trial]] * len(all_pressures)}
            indiv = pd.DataFrame(indiv)

            all_data = all_data.append(indiv)

    compressed_pickle('../processed_data/regional_pressure_validation', all_data)

    return all_data

def pressure_alignment_on_foot():
    """ Plots mean pressure across all trials for each participant, with standardized foot overlaid to demonstrate
    mapping across all participants

    Returns:
            Figure

    """
    x_outline, y_outline = get_foot_outline()

    k = 0
    plt.figure(figsize=(10, 20), dpi=600)
    for participant in participant_ids:

        foot = 'left'
        trial = 'trial07'
        file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
        data = decompress_pickle(file)

        ppt_data = np.zeros((data['Raw data'].shape[0], data['Raw data'].shape[1], 0))

        if calibration_type[participant] == 'point':
            continue
        else:

            k += 1

            for trial in trial_ids[5:-1]:
                if participant not in details[trial]:
                    continue
                else:

                    foot = 'left'

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    raw = np.mean(data['Raw data'], axis=2)

                    ppt_data = np.dstack((ppt_data, raw))

        raw = np.mean(ppt_data, axis=2)

        plt.subplot(4, 4, k)
        plt.title(participant)
        plt.imshow(raw, cmap='viridis', extent=[x_outline.min(), x_outline.max(), y_outline.min(), y_outline.max()])
        plt.scatter(x_outline, y_outline, c='white', s=2.)
        plt.axis('off')
    plt.savefig('../individual_figures/pressure_alignment.png', dpi=600)

def proportion_mapped_sensors():

    x_outline, y_outline = get_foot_outline()

    proportion_mapped = {}

    q=0
    plt.figure(figsize=(25, 20))
    for foot in feet:

        proportion_mapped[foot] = {}

        for participant in participant_ids:

            if calibration_type[participant] == 'point':
                continue
            else:
                q+=1

                proportion_mapped[foot][participant] = {}

                file = filepath_prefix + participant + '/trial01/' + foot + '/' + foot + ' data.pbz2'
                data = decompress_pickle(file)

                concated = np.zeros((data['Raw data'].shape[0], data['Raw data'].shape[1]))

                for trial in trial_ids:

                    if participant not in details[trial]:
                        continue
                    else:

                        file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                        data = decompress_pickle(file)

                        concated = np.dstack((concated, data['Raw data']))

                D = concated
                reshaped = np.zeros((0, D.shape[2]))
                for i in range(D.shape[1]):

                    for j in range(D.shape[0]):
                        reshaped = np.vstack((reshaped, D[j, i, :]))


                dim = D.shape  # Dimensions
                cmin = [min(x_outline), min(y_outline)]
                cmax = [max(x_outline), max(y_outline)]

                c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
                c1 = np.linspace(cmin[0], cmax[0], dim[1])

                print('outline achieved')
                #dim = D.shape  # Dimensions
                #cmin = np.min(fs.foot_surface.bbox_min, axis=0)  # Calculates bounding box for arbitrary boundary. # y
                #cmax = np.max(fs.foot_surface.bbox_max, axis=0)  # x


                #c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
                #c1 = np.linspace(cmin[0], cmax[0], dim[1])

                blue = 0
                green = 0
                red = 0
                orange = 0

                pressure_mapped = np.zeros((0, dim[2]))
                pressure_not_mapped = np.zeros((0, dim[2]))

                D[D < 1] = 0.0
                #plt.figure()
                plt.subplot(4, 8, q)
                plt.axis('off')
                plt.scatter(x_outline, y_outline, c='black', s=3.)
                for i in range(dim[1]):

                    for j in range(dim[0]):

                        #loca = fs.foot_surface.pixel2hand(np.array([c1[i], c0[j]]))
                        loca = np.array([c1[i], c0[j]])

                        if (np.isnan(D[j, i, 0]) is False or np.nanmax(D[j, i, :]) > 0.0) and \
                                fs.foot_surface.locate(loca)[0][
                                    0] == '':
                            plt.scatter(loca[0], loca[1], s=5, c='blue', label='data but no region', marker=',')
                            red += 1
                            pressure_not_mapped = np.vstack((pressure_not_mapped, D[j, i, :]))

                        if (np.isnan(D[j, i, 0]) is False or np.nanmax(D[j, i, :]) > 0.0) and \
                                fs.foot_surface.locate(loca)[0][
                                    0] != '':
                            plt.scatter(loca[0], loca[1], s=5, c='green', label='data and region', marker=',')
                            green += 1
                            pressure_mapped = np.vstack((pressure_mapped, D[j, i, :]))

                        if (np.isnan(D[j, i, 0]) or np.nanmax(D[j, i, :]) < 1.0) and fs.foot_surface.locate(loca)[0][
                            0] == '':
                            plt.scatter(loca[0], loca[1], s=5, c='red', label='no data and no region', marker=',')
                            blue += 1

                        if (np.isnan(D[j, i, 0]) or np.nanmax(D[j, i, :]) < 1.0) and fs.foot_surface.locate(loca)[0][
                            0] != '':
                            plt.scatter(loca[0], loca[1], s=5, c='orange', label='no data and region', marker=',')
                            orange += 1


                print('Active off foot: ', red)
                print('Active on foot: ', green)
                print('Not active off foot: ', blue)
                print('Not active on foot: ', orange)
                print('Proportion active mapped: ', green / (red + green) * 100)
                print(np.nanmean((np.sum(pressure_mapped,axis=0) / (np.sum(pressure_mapped,axis=0) + np.sum(pressure_not_mapped,axis=0))) * 100))
                #print(np.sum(pressure_mapped) / (np.sum(pressure_mapped) + np.sum(pressure_not_mapped)))


                #plt.axis("off")
                #plt.show()
                proportion_mapped[foot][participant]['Active off foot'] = red
                proportion_mapped[foot][participant]['Active on foot'] = green
                proportion_mapped[foot][participant]['Not active off foot'] = blue
                proportion_mapped[foot][participant]['Not active on foot'] = orange
                proportion_mapped[foot][participant]['Proportion active mapped'] = green / (blue + green) * 100
                proportion_mapped[foot][participant]['Proportion pressure mapped'] = np.nanmean((np.sum(pressure_mapped,
                                                                                                        axis=0) / (
                                                                                                             np.sum(
                                                                                                                 pressure_mapped,
                                                                                                                 axis=0) + np.sum(
                                                                                                         pressure_not_mapped,
                                                                                                         axis=0))) * 100)

    plt.savefig('../individual_figures/mapping_all.png')
    compressed_pickle(
        '../processed_data/proportion_mapped_sensors', \
        proportion_mapped)
