from insole import *
from scipy import stats
import numpy as np
import pandas as pd
from project_insole_constants import *

def force_area_correlation(step_all_metrics, all_metrics):
    """ Figure containing 2D histograms for the relationship between force and contact are.
        Also runs correlation on the data and prints R-squared and p-value

    Args:
        all_metrics: dataframe containing analysed data for all tasks
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:

    """

    all_metrics = all_metrics[all_metrics['Force'] > 5.]
    minimal = all_metrics[all_metrics['Trial type'] == 'Standing']
    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping']

    min_shapiro_area = stats.shapiro(minimal['Contact area percent'])
    min_shapiro_pres = stats.shapiro(minimal['Force'])
    min_corr = stats.spearmanr(minimal['Contact area percent'], minimal['Force'])

    print('area: ', min_shapiro_area)
    print('pressure: ', min_shapiro_pres)
    print('R2: ', min_corr[0] ** 2)
    print('p: ', min_corr[1])
    print('df: ', minimal['Contact area percent'].shape[0] - 2)

    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]
    step_shapiro_area = stats.shapiro(step_all_metrics['Contact area percent'])
    step_shapiro_pres = stats.shapiro(step_all_metrics['Force'])
    step_corr = stats.spearmanr(step_all_metrics['Contact area percent'], step_all_metrics['Force'])

    print('area: ', step_shapiro_area)
    print('pressure: ', step_shapiro_pres)
    print('R2: ', step_corr[0] ** 2)
    print('p: ', step_corr[1])
    print('df: ', step_all_metrics['Contact area percent'].shape[0] - 2)

    jump_shapiro_area = stats.shapiro(jumping['Contact area percent'])
    jump_shapiro_pres = stats.shapiro(jumping['Force'])
    jump_corr = stats.spearmanr(jumping['Contact area percent'], jumping['Force'])

    print('area: ', jump_shapiro_area)
    print('pressure: ', jump_shapiro_pres)
    print('R2: ', jump_corr[0] ** 2)
    print('p: ', jump_corr[1])
    print('df: ', jumping['Contact area percent'].shape[0] - 2)



def CoP_CoC_correlation(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates):
    """ Calculate R-squared showing shared variance between the CoP and CoC coordinates along both axis

    Args:
        all_metrics: dataframe containing analysed data for all tasks
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths
        all_CoP_coordinates: dataframe containing CoP locations at all timepoints for all tasks
        step_CoP_coordinates: dataframe containing CoP locations at all timepoints for walking trials calculated using
            normalized steps

    Returns:

    """

    x_outline, y_outline = get_foot_outline()

    all_CoP_coordinates = all_CoP_coordinates[all_CoP_coordinates['Trial type'] != 'Locomotion']
    all_metrics = all_metrics[all_metrics['Trial type'] != 'Locomotion']

    all_CoP_coordinates['X_distance'] = np.abs((all_CoP_coordinates['X_contact'] - all_CoP_coordinates['X_pressure'])) / (x_outline.max() - x_outline.min()) * 100
    all_CoP_coordinates['Y_distance'] = np.abs((all_CoP_coordinates['Y_contact'] - all_CoP_coordinates['Y_pressure'])) / (y_outline.max() - y_outline.min()) * 100


    all_CoP_coordinates = all_CoP_coordinates.reset_index()
    all_metrics = all_metrics.reset_index()

    forces = all_metrics['Force'].copy()
    all_CoP_coordinates['Force'] = forces.values

    all_CoP_coordinates = all_CoP_coordinates[all_CoP_coordinates['Force'] > 5.]


    ### locomotion trials

    step_CoP_coordinates['X_distance'] = np.abs((step_CoP_coordinates['X_contact'] - step_CoP_coordinates['X_pressure'])) / (x_outline.max() - x_outline.min()) * 100
    step_CoP_coordinates['Y_distance'] = np.abs((step_CoP_coordinates['Y_contact'] - step_CoP_coordinates['Y_pressure'])) / (y_outline.max() - y_outline.min()) * 100

    step_CoP_coordinates = step_CoP_coordinates.reset_index()
    step_all_metrics = step_all_metrics.reset_index()

    forces = step_all_metrics['Force'].copy()
    step_CoP_coordinates['Force'] = forces.values

    step_CoP_coordinates = step_CoP_coordinates[step_CoP_coordinates['Force'] > 5.]


    distance_df = pd.concat([step_CoP_coordinates, all_CoP_coordinates])

    print('All:')
    print('X R2: ', stats.pearsonr(distance_df['X_pressure'], distance_df['X_contact'])[0]**2)
    print('Y R2: ', stats.pearsonr(distance_df['Y_pressure'], distance_df['Y_contact'])[0]**2)


    for trial_type in pd.unique(distance_df['Trial type']):
        trial_df = distance_df[distance_df['Trial type'] == trial_type]

        print(trial_type)
        print('X R2: ', stats.pearsonr(trial_df['X_pressure'], trial_df['X_contact'])[0]**2)
        print('Y R2: ', stats.pearsonr(trial_df['Y_pressure'], trial_df['Y_contact'])[0]**2)


    for trial in pd.unique(distance_df['Trial']):
        trial_df = distance_df[distance_df['Trial'] == trial]

        print(trial)
        print('X R2: ', stats.pearsonr(trial_df['X_pressure'], trial_df['X_contact'])[0]**2)
        print('Y R2: ', stats.pearsonr(trial_df['Y_pressure'], trial_df['Y_contact'])[0]**2)



def compare_mean_area(all_metrics, step_all_metrics):
    # all trials
    all_metrics = all_metrics[all_metrics['Force'] > 5.]

    # locomotion trials
    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]

    # Extract standing and jumping trials
    standing = all_metrics[all_metrics['Trial type'] == 'Standing']
    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping']


    ## Extract individual trials

    # standing trials
    trial5 = standing[standing['Trials'] == 'trial05']
    trial6 = standing[standing['Trials'] == 'trial06']
    trial12 = standing[standing['Trials'] == 'trial12']
    trial10 = standing[standing['Trials'] == 'trial10']
    trial11 = standing[standing['Trials'] == 'trial11']

    # jumping trials
    trial16 = jumping[jumping['Trials'] == 'trial16']
    trial17 = jumping[jumping['Trials'] == 'trial17']
    trial18 = jumping[jumping['Trials'] == 'trial18']

    # locomotion trials
    trial9 = step_all_metrics[step_all_metrics['Trials'] == 'trial09']
    trial7 = step_all_metrics[step_all_metrics['Trials'] == 'trial07']
    trial8 = step_all_metrics[step_all_metrics['Trials'] == 'trial08']
    trial15 = step_all_metrics[step_all_metrics['Trials'] == 'trial15']
    trial13 = step_all_metrics[step_all_metrics['Trials'] == 'trial13']
    trial14 = step_all_metrics[step_all_metrics['Trials'] == 'trial14']
    trial19 = step_all_metrics[step_all_metrics['Trials'] == 'trial19']

    ## Define post hoc comparisons
    standing_comparisons = {'trial05': ['trial06', 'trial12', 'trial10', 'trial11'],
                            'trial06': ['trial12', 'trial10', 'trial11'],
                            'trial12': ['trial10', 'trial11'],
                            'trial10': ['trial11']}

    jumping_comparisons = {'trial16': ['trial17', 'trial18'],
                           'trial17': ['trial18']}

    locomotion_comparisons = {'trial09': ['trial07', 'trial08', 'trial15', 'trial13', 'trial14', 'trial19'],
                              'trial07': ['trial08', 'trial15', 'trial13', 'trial14', 'trial19'],
                              'trial08': ['trial15', 'trial13', 'trial14', 'trial19'],
                              'trial15': ['trial13', 'trial14', 'trial19'],
                              'trial13': ['trial14', 'trial19'],
                              'trial14': ['trial19']}

    ## Run one-way ANOVAs

    # Standing

    print('Force, standing')
    # test of homogeneity of variance (Levene's)
    print(stats.levene(trial5['Force'], trial6['Force'], trial12['Force'], trial10['Force'], trial11['Force']))

    # test of normality (Shapiro-Wilk)
    print(stats.shapiro(standing['Force']))

    # non-parametric ANOVA
    print(stats.kruskal(trial5['Force'], trial6['Force'], trial12['Force'], trial10['Force'], trial11['Force']))

    # post-hoc tests (Mann-Whiteney U)
    for compare_to in standing_comparisons:
        for compare_with in standing_comparisons[compare_to]:
            print(trial_titles[compare_to], trial_titles[compare_with], '\t:', \
                  stats.mannwhitneyu(standing[standing['Trials'] == compare_to]['Force'], \
                                     standing[standing['Trials'] == compare_with]['Force']))

    # Jumping

    print('Force, jumping')
    # test of homogeneity of variance (Levene's)
    print(stats.levene(trial16['Force'], trial17['Force'], trial18['Force']))

    # test of normality (Shapiro-Wilk)
    print(stats.shapiro(jumping['Force']))

    # non-parametric ANOVA
    print(stats.kruskal(trial16['Force'], trial17['Force'], trial18['Force']))

    # post-hoc tests (Mann-Whiteney U)
    for compare_to in jumping_comparisons:
        for compare_with in jumping_comparisons[compare_to]:
            print(trial_titles[compare_to], trial_titles[compare_with], \
                  '\t:', stats.mannwhitneyu(jumping[jumping['Trials'] == compare_to]['Force'], \
                                            jumping[jumping['Trials'] == compare_with]['Force']))


    # Locomotion
    print('Force, locomotion')
    # test of homogeneity of variance (Levene's)
    print(stats.levene(trial9['Force'], trial7['Force'], trial8['Force'], trial15['Force'], trial13['Force'],\
                trial14['Force'], trial19['Force']))

    # test of normality (Shapiro-Wilk)
    print(stats.shapiro(step_all_metrics['Force']))

    # non-parametric ANOVA
    print(stats.kruskal(trial9['Force'], trial7['Force'], trial8['Force'], trial15['Force'], trial13['Force'],\
                trial14['Force'], trial19['Force']))

    # post-hoc tests (Mann-Whiteney U)
    for compare_to in locomotion_comparisons:
        for compare_with in locomotion_comparisons[compare_to]:
            print(trial_titles[compare_to], trial_titles[compare_with], '\t:', \
                  stats.mannwhitneyu(step_all_metrics[step_all_metrics['Trials'] == compare_to]['Force'], \
                                     step_all_metrics[step_all_metrics['Trials'] == compare_with]['Force']))

    # Standing

    print('Area, standing')
    # test of homogeneity of variance (Levene's)
    print(stats.levene(trial5['Contact area percent'], trial6['Contact area percent'], trial12['Contact area percent'], trial10['Contact area percent'], trial11['Contact area percent']))

    # test of normality (Shapiro-Wilk)
    print(stats.shapiro(standing['Contact area percent']))

    # non-parametric ANOVA
    print(stats.kruskal(trial5['Contact area percent'], trial6['Contact area percent'], trial12['Contact area percent'], trial10['Contact area percent'], trial11['Contact area percent']))

    # post-hoc tests (Mann-Whiteney U)
    for compare_to in standing_comparisons:
        for compare_with in standing_comparisons[compare_to]:
            print(trial_titles[compare_to], trial_titles[compare_with], '\t:', \
                  stats.mannwhitneyu(standing[standing['Trials'] == compare_to]['Contact area percent'], \
                                     standing[standing['Trials'] == compare_with]['Contact area percent']))

    # Jumping

    print('Area, jumping')
    # test of homogeneity of variance (Levene's)
    print(stats.levene(trial16['Contact area percent'], trial17['Contact area percent'], trial18['Contact area percent']))

    # test of normality (Shapiro-Wilk)
    print(stats.shapiro(jumping['Contact area percent']))

    # non-parametric ANOVA
    print(stats.kruskal(trial16['Contact area percent'], trial17['Contact area percent'], trial18['Contact area percent']))

    # post-hoc tests (Mann-Whiteney U)
    for compare_to in jumping_comparisons:
        for compare_with in jumping_comparisons[compare_to]:
            print(trial_titles[compare_to], trial_titles[compare_with], \
                  '\t:', stats.mannwhitneyu(jumping[jumping['Trials'] == compare_to]['Contact area percent'], \
                                            jumping[jumping['Trials'] == compare_with]['Contact area percent']))

    # Locomotion
    print('Area, locomotion')
    # test of homogeneity of variance (Levene's)
    print(stats.levene(trial9['Contact area percent'], trial7['Contact area percent'], trial8['Contact area percent'], trial15['Contact area percent'], trial13['Contact area percent'], \
                       trial14['Contact area percent'], trial19['Contact area percent']))

    # test of normality (Shapiro-Wilk)
    print(stats.shapiro(step_all_metrics['Contact area percent']))

    # non-parametric ANOVA
    print(stats.kruskal(trial9['Contact area percent'], trial7['Contact area percent'], trial8['Contact area percent'], trial15['Contact area percent'], trial13['Contact area percent'], \
                        trial14['Contact area percent'], trial19['Contact area percent']))

    # post-hoc tests (Mann-Whiteney U)
    for compare_to in locomotion_comparisons:
        for compare_with in locomotion_comparisons[compare_to]:
            print(trial_titles[compare_to], trial_titles[compare_with], '\t:', \
                  stats.mannwhitneyu(step_all_metrics[step_all_metrics['Trials'] == compare_to]['Contact area percent'], \
                                     step_all_metrics[step_all_metrics['Trials'] == compare_with]['Contact area percent']))



def mean_std_force_area(all_metrics, step_all_metrics):

    # all trials
    all_metrics = all_metrics[all_metrics['Force'] > 5.]


    all_metrics.groupby('Trial title').describe().to_csv('../descriptive_statistic_csvs/all_task_descriptives.csv')
    all_metrics.groupby('Trial type').describe().to_csv('../descriptive_statistic_csvs/task_type_descriptives.csv')
    all_metrics.groupby(['Trial title', 'Participant']).describe().to_csv(
        '../descriptive_statistic_csvs/task_type_participant_descriptives.csv')

    # locomotion trials
    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]

    step_all_metrics.groupby('Trial title').describe().to_csv('../descriptive_statistic_csvs/step_task_descriptives.csv')

    step_all_metrics.groupby('Trial type').describe().to_csv(
        '../descriptive_statistic_csvs/task_type_step_descriptives.csv')