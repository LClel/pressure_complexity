import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from project_insole_constants import *
from project_insole_analysis_information import *
from step_processing_functions import *
import footsim as fs
from footsim.surface import *
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from scipy.spatial import ConvexHull
from scipy import stats
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mycolorpy import colorlist as mcp
import random
import scipy
import scipy.cluster.hierarchy as sch
from insole import *
import pickle as pk
import numpy as np
import random
import scipy
import scipy.cluster.hierarchy as sch

sns.set_style("white")

def pressure_overview(all_metrics, step_all_metrics):
    """ This figure contains
    Top: Distribution of force experienced between task types
    Middle: 3 panels containing force experienced in each task within each task type
    Bottom: Force experienced when the centre of pressure is in each of the 4 coarse regions of the foot

    Args:
        all_metrics: dataframe containing analysed data for all tasks
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:
        Figure
    """
    # generate figure
    fig = plt.figure(constrained_layout=True, figsize=(24, 30), dpi=100)
    #plt.rcParams.update({'font.size': 18})
    gs = GridSpec(3, 3, figure=fig, height_ratios=[.85, .85, .85])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, :])

    all_metrics = all_metrics[all_metrics['Force'] > 5.] # remove data that relates to extremely low forces
    standing = all_metrics[all_metrics['Trial type'] == 'Standing'] # extract only standing tasks
    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping'] # extract only jumping tasks

    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.] # remove data that relates to extremely low forces
    step_all_metrics = step_all_metrics.drop(columns='Step number') # remove column

    all_tasks = pd.concat([standing, step_all_metrics, jumping]) # joing dataframes together

    plt.suptitle('Force', fontsize=60)
    sns.set_palette(trial_classification_colour_palette)
    kde1 = sns.kdeplot(data=all_tasks, x='Force', hue='Trial type', fill=True, alpha=.5,
                       ax=ax1, legend=False, common_norm=False, linewidth=0., hue_order=['Standing', 'Locomotion', 'Jumping'])
    kde1.legend(title="Task type", fontsize=35, loc='upper right', labels=['Standing', 'Locomotion', 'Jumping'])
    ax1.set_xlim(0, 300)
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.tick_params(axis='x', labelsize=40)
    ax1.set_xlabel('Force', fontsize=45)
    sns.despine(ax=ax1)

    sns.set_palette(standing_palette)
    ax2.set_title('Standing', fontsize=55)
    kde2 = sns.kdeplot(data=standing, x='Force', alpha=.5, hue='Trial title', fill=False, ax=ax2,
                       common_norm=False, linewidth=3.5, hue_order=['Quiet standing', 'Wobble-board', 'Twisting','Sit-to-stand','Sit-to-walk'])
    ax2.set_xlim(0, 125)
    ax2.set_yticks([])
    ax2.set_ylabel('')
    ax2.tick_params(axis='x', labelsize=40)
    kde2.legend(title="Task", fontsize=35, loc='upper right', labels=['Sit-to-walk', 'Sit-to-stand', 'Twisting', 'Wobble-board', 'Standing'])
    ax2.set_xlabel('Force', fontsize=45)
    sns.despine(ax=ax2)

    sns.set_palette(locomotion_palette)
    ax3.set_title('Locomotion', fontsize=55)
    kde3 = sns.kdeplot(data=step_all_metrics, x='Force', alpha=.5, hue='Trial title', fill=False, ax=ax3,
                       common_norm=False, linewidth=3.5, hue_order=['Normal walking', 'Fast walking', 'Slow walking', 'Jogging', 'Gravel walking', 'Stairs','Slope walking'])
    ax3.set_xlim(0, 350)
    ax3.set_yticks([])
    ax3.set_ylabel('')
    ax3.tick_params(axis='x', labelsize=40)
    ax3.legend(fontsize=35)
    kde3.legend(title="Task", fontsize='20', loc='upper right', labels=['Slope walking','Stairs','Gravel walking', 'Jogging', 'Slow walking', 'Fast walking', 'Normal walking'])
    ax3.set_xlabel('Force', fontsize=45)
    sns.despine(ax=ax3)

    sns.set_palette(jumping_palette)
    ax4.set_title('Jumping', fontsize=55)
    kde4 = sns.kdeplot(data=jumping, x='Force', alpha=.5, hue='Trial title', fill=False, ax=ax4,
                       common_norm=False, linewidth=3.5,
                       hue_order=['Both feet jump','Left foot jump','Right foot jump'])
    ax4.set_xlim(0, 550)
    ax4.set_yticks([])
    ax4.set_ylabel('')
    ax4.tick_params(axis='x', labelsize=40)
    ax4.legend(fontsize=35)
    kde4.legend(title="Task", fontsize='20', loc='upper right',
                labels=['Right foot jump','Left foot jump','Both feet jump'])
    ax4.set_xlabel('Force', fontsize=55)
    sns.despine(ax=ax4)

    sns.set_palette(regions_color_palette)
    ax5.set_title('Force when centre of pressure is in each region', fontsize=60)
    kde5 = sns.kdeplot(data=all_tasks, x='Force', alpha=.5, hue='CoP location', fill=True, ax=ax5,
                       common_norm=False, linewidth=0.,
                       hue_order=['H', 'A', 'M', 'T'])
    ax5.set_xlim(0, 300)
    ax5.set_yticks([])
    ax5.set_ylabel('')
    ax5.tick_params(axis='x', labelsize=40)
    ax5.legend(fontsize=35)
    kde5.legend(title='CoP location', fontsize='20', loc='upper right',
                labels=['Toes', 'Metatarsals', 'Arch', 'Heel'])
    ax5.set_xlabel('Force', fontsize=45)
    sns.despine(ax=ax5)

    plt.tight_layout()

    plt.savefig('../individual_figures/pressure_panel.png')

def contact_area_overview(all_metrics, step_all_metrics):
    """ This figure contains
        Top: Distribution of contact area experienced between task types
        Middle: 3 panels containing contact area experienced in each task within each task type
        Bottom: Contact area experienced when the centre of pressure is in each of the 4 coarse regions of the foot

        Args:
            all_metrics: dataframe containing analysed data for all tasks
            step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

        Returns:
            Figure
        """
    fig = plt.figure(constrained_layout=True, figsize=(24, 30), dpi=100)
    plt.rcParams.update({'font.size': 18})
    gs = GridSpec(3, 3, figure=fig, height_ratios=[.85, .85, .85])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, :])

    all_metrics = all_metrics[all_metrics['Force'] > 5.]  # remove data that relates to extremely low forces
    standing = all_metrics[all_metrics['Trial type'] == 'Standing']  # extract only standing tasks
    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping']  # extract only jumping tasks

    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]  # remove data that relates to extremely low forces
    step_all_metrics = step_all_metrics.drop(columns='Step number')  # remove column

    all_tasks = pd.concat([standing, step_all_metrics, jumping])  # joing dataframes together


    plt.suptitle('Contact area', fontsize=60)
    sns.set_palette(trial_classification_colour_palette)
    kde1 = sns.kdeplot(data=all_tasks, x='Contact area percent', hue='Trial type', fill=True, alpha=.5,
                       ax=ax1, legend=False, common_norm=False, linewidth=0.,
                       hue_order=['Standing', 'Locomotion', 'Jumping'])
    kde1.legend(title="Task type", fontsize=35, loc='upper right', labels=['Locomotion', 'Standing', 'Jumping'])
    ax1.set_xlim(0, 100)
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.tick_params(axis='x', labelsize=40)
    ax1.set_xlabel('Contact area', fontsize=55)
    sns.despine(ax=ax1)

    sns.set_palette(standing_palette)
    ax2.set_title('Standing', fontsize=55)
    kde2 = sns.kdeplot(data=standing, x='Contact area percent', alpha=.5, hue='Trial title', fill=False,
                       ax=ax2,
                       common_norm=False, linewidth=3.5,
                       hue_order=['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk'])
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_ylabel('')
    ax2.tick_params(axis='x', labelsize=40)
    kde2.legend(title="Task", fontsize=35, loc='upper right',
                labels=['Sit-to-walk', 'Sit-to-stand', 'Twisting', 'Wobble-board', 'Quiet standing'])
    # ax2.legend()
    ax2.set_xlabel('Contact area', fontsize=45)
    sns.despine(ax=ax2)

    sns.set_palette(locomotion_palette)
    ax3.set_title('Locomotion', fontsize=55)
    kde3 = sns.kdeplot(data=step_all_metrics, x='Contact area percent', alpha=.5, hue='Trial title', fill=False, ax=ax3,
                       common_norm=False, linewidth=3.5,
                       hue_order=['Normal walking', 'Fast walking', 'Slow walking', 'Jogging', 'Gravel walking', 'Stairs', 'Slope walking'])
    ax3.set_xlim(0, 100)
    ax3.set_yticks([])
    ax3.set_ylabel('')
    ax3.tick_params(axis='x', labelsize=40)
    ax3.legend(fontsize=35)
    kde3.legend(title="Task", fontsize=35, loc='upper right',
                labels=['Slope walking', 'Stairs', 'Gravel walking', 'Jogging', 'Slow walking', 'Fast walking', 'Normal walking'])
    ax3.set_xlabel('Contact area', fontsize=55)
    sns.despine(ax=ax3)

    sns.set_palette(jumping_palette)
    ax4.set_title('Jumping', fontsize=55)
    kde4 = sns.kdeplot(data=jumping, x='Contact area percent', alpha=.5, hue='Trial title', fill=False, ax=ax4,
                       common_norm=False, linewidth=3.5,
                       hue_order=['Both feet jump', 'Left foot jump', 'Right foot jump'])
    ax4.set_xlim(0, 100)
    ax4.set_yticks([])
    ax4.set_ylabel('')
    ax4.tick_params(axis='x', labelsize=40)
    ax4.legend(fontsize=35)
    kde4.legend(title="Task", fontsize=35, loc='upper right',
                labels=['Right foot jump', 'Left foot jump', 'Both feet jump'])
    ax4.set_xlabel('Contact area', fontsize=45)
    sns.despine(ax=ax4)

    sns.set_palette(regions_color_palette)
    ax5.set_title('Contact area when centre of pressure is in each region', fontsize=60)
    kde5 = sns.kdeplot(data=all_tasks, x='Contact area percent', alpha=.5, hue='CoP location', fill=True, ax=ax5,
                       common_norm=False, linewidth=0.,
                       hue_order=['H', 'A', 'M', 'T'])
    ax5.set_xlim(0, 100)
    ax5.set_yticks([])
    ax5.set_ylabel('')
    ax5.tick_params(axis='x', labelsize=40)
    ax5.legend(fontsize=35)
    kde5.legend(title="CoP location", fontsize=35, loc='upper right',
                labels=['Toes', 'Metatarsals', 'Arch', 'Heel'])
    ax5.set_xlabel('Contact area', fontsize=45)
    sns.despine(ax=ax5)

    plt.tight_layout()

    plt.savefig('../individual_figures/contact_area_panel.png')

def split_data_between_trials():
    plt.rcParams.update({'font.size': 30})
    tot = 278626 + 644856 + 42550
    df = pd.DataFrame({'Trial type': ['Standing','Walking','Jumping'],
               'Amount of data': [(278626 / tot) * 100, (644856 / tot) * 100, (42550 / tot) * 100]})

    df_plot = df.groupby(['Trial type', 'Amount of data']).size().reset_index().pivot(columns='Amount of data', index='Trial type', values=0)

    df_plot.plot(kind='bar',stacked=True,color=['darkviolet', 'springgreen', 'red'])
    plt.savefig('../individual_figures/data separation.png')


def CoP_traces(normalized_steps):
    """ Generates centre of pressure traces for PPT 004 in selected trials:
        - quiet standing
        - wobble-board
        - slow walking
        - normal walking
        - fast walking
        - gravel walking

    Args:
        normalized_steps (dict): dictionary containing pressure data for each step in each walking task
            for each participant

    Returns:

    """
    x_outline, y_outline = get_foot_outline()

    df = pd.read_csv('../processed_data/CoP_coordinates_df.csv')

    df['X'] = df['X'] / (np.max(x_outline) - np.min(x_outline))
    df['Y'] = df['Y'] / (np.max(y_outline) - np.min(y_outline))


    fig = plt.figure(constrained_layout=True, figsize=(12, 6), dpi=100)
    plt.rcParams.update({'font.size': 30})
    gs = GridSpec(6, 13, figure=fig, wspace=.15)

    ax2 = fig.add_subplot(gs[:2, 1])
    ax3 = fig.add_subplot(gs[:2, 2])
    ax4 = fig.add_subplot(gs[:2, 4])
    ax5 = fig.add_subplot(gs[:2, 5])
    ax6 = fig.add_subplot(gs[2:4, 1])
    ax7 = fig.add_subplot(gs[2:4, 2])
    ax8 = fig.add_subplot(gs[2:4, 4])
    ax9 = fig.add_subplot(gs[2:4, 5])
    ax10 = fig.add_subplot(gs[2:4, 7])
    ax11 = fig.add_subplot(gs[2:4, 8])
    ax12 = fig.add_subplot(gs[2:4, 10])
    ax13 = fig.add_subplot(gs[2:4, 11])

    ax2.scatter(x_outline, y_outline, s=1, c='black')
    ax2.axis('off')

    ax3.scatter(x_outline, y_outline, s=1, c='black')
    ax3.axis('off')

    ax4.scatter(x_outline, y_outline, s=1, c='black')
    ax4.axis('off')

    ax5.scatter(x_outline, y_outline, s=1, c='black')
    ax5.axis('off')

    ax6.scatter(x_outline, y_outline, s=1, c='black')
    ax6.axis('off')

    ax7.scatter(x_outline, y_outline, s=1, c='black')
    ax7.axis('off')

    ax8.scatter(x_outline, y_outline, s=1, c='black')
    ax8.axis('off')

    ax9.scatter(x_outline, y_outline, s=1, c='black')
    ax9.axis('off')

    ax10.scatter(x_outline, y_outline, s=1, c='black')
    ax10.axis('off')

    ax11.scatter(x_outline, y_outline, s=1, c='black')
    ax11.axis('off')

    ax12.scatter(x_outline, y_outline, s=1, c='black')
    ax12.axis('off')

    ax13.scatter(x_outline, y_outline, s=1, c='black')
    ax13.axis('off')


    ####

    selected_trials = ['trial09', 'trial07', 'trial08', 'trial19']
    participants = ['PPT_004']
    all_CoPs = {}

    i=0
    for trial in selected_trials:
        all_CoPs[trial] = {}

        for foot in feet:


            trial_CoP = np.zeros((0, 2))

            for participant in participants:
                # for participant in participants:

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

                        loc, x, y = centre_of_pressure(D, threshold=250)

                        colors = mcp.gen_color_normalized(cmap='spring', data_arr=list(range(100)))


                        trial_CoP = np.vstack((trial_CoP, loc))

            all_CoPs[trial][foot] = trial_CoP

        i += 1


    for i in range(int(len(all_CoPs['trial09']['left'][:,1])/100)-2):

        ax6.scatter(all_CoPs['trial09']['left'][(100*i):((100*i)+100),0], all_CoPs['trial09']['left'][(100*i):((100*i)+100),1], c=colors, s=2.)

        ax7.scatter(all_CoPs['trial09']['right'][(100*i):((100*i)+100),0], all_CoPs['trial09']['right'][(100*i):((100*i)+100),1], c=colors, s=2.)
    ax7.invert_xaxis()


    for i in range(int(len(all_CoPs['trial07']['left'][:,1])/100)-2):

        ax8.scatter(all_CoPs['trial07']['left'][(100*i):((100*i)+100),0], all_CoPs['trial07']['left'][(100*i):((100*i)+100),1], c=colors, s=2.)

        ax9.scatter(all_CoPs['trial07']['right'][(100*i):((100*i)+100),0], all_CoPs['trial07']['right'][(100*i):((100*i)+100),1], c=colors, s=2.)
    ax9.invert_xaxis()


    for i in range(int(len(all_CoPs['trial08']['left'][:,1])/100)-2):

        ax10.scatter(all_CoPs['trial08']['left'][(100*i):((100*i)+100),0], all_CoPs['trial08']['left'][(100*i):((100*i)+100),1], c=colors, s=2.)

        ax11.scatter(all_CoPs['trial08']['right'][(100*i):((100*i)+100),0], all_CoPs['trial08']['right'][(100*i):((100*i)+100),1], c=colors, s=2.)
    ax11.invert_xaxis()



    for i in range(int(len(all_CoPs['trial19']['left'][:,1])/100)-2):

        ax12.scatter(all_CoPs['trial19']['left'][(100*i):((100*i)+100),0], all_CoPs['trial19']['left'][(100*i):((100*i)+100),1], c=colors, s=2.)

        ax13.scatter(all_CoPs['trial19']['right'][(100*i):((100*i)+100),0], all_CoPs['trial19']['right'][(100*i):((100*i)+100),1], c=colors, s=2.)
    ax13.invert_xaxis()

    selected_trials = ['trial05', 'trial06']
    participants = ['PPT_004']
    all_CoPs = {}

    filepath_prefix = '../preprocessed_data/'

    for trial in selected_trials:
        all_CoPs[trial] = {}

        for foot in feet:
            all_CoPs[trial][foot] = {}

            trial_CoP = np.zeros((0, 2))

            for participant in participants:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    # load in the data for the participant
                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    D = data['Raw data']

                    loc, x, y = centre_of_pressure(D, threshold=250)

                    colors = mcp.gen_color_normalized(cmap='spring', data_arr=list(range(D.shape[2])))

                    trial_CoP = np.vstack((trial_CoP, loc))


                all_CoPs[trial][foot]['data'] = trial_CoP
                all_CoPs[trial][foot]['colours'] = colors


    ax2.scatter(all_CoPs['trial05']['left']['data'][:,0],
                all_CoPs['trial05']['left']['data'][:,1], c=all_CoPs['trial05']['left']['colours'], s=2.)

    ax3.scatter(all_CoPs['trial05']['right']['data'][:,0],
                all_CoPs['trial05']['right']['data'][:,1], c=all_CoPs['trial05']['right']['colours'], s=2.)
    ax3.invert_xaxis()

    ax4.scatter(all_CoPs['trial06']['left']['data'][:,0],
                all_CoPs['trial06']['left']['data'][:,1], c=all_CoPs['trial06']['left']['colours'], s=2.)

    ax5.scatter(all_CoPs['trial06']['right']['data'][:,0],
                all_CoPs['trial06']['right']['data'][:,1], c=all_CoPs['trial06']['right']['colours'], s=2.)
    ax5.invert_xaxis()

    plt.savefig('../individual_figures/CoP_on_foot.png', dpi=100)


def CoC_traces(normalized_steps):
    """ Generates centre of contact traces for PPT 004 in selected trials:
        - quiet standing
        - wobble-board
        - slow walking
        - normal walking
        - fast walking
        - gravel walking

    Args:
        normalized_steps (dict): dictionary containing pressure data for each step in each walking task
            for each participant

    Returns:

    """

    x_outline, y_outline = get_foot_outline()


    fig = plt.figure(constrained_layout=True, figsize=(12, 6), dpi=100)
    plt.rcParams.update({'font.size': 30})
    gs = GridSpec(6, 13, figure=fig, wspace=.15)

    ax2 = fig.add_subplot(gs[:2, 1])
    ax3 = fig.add_subplot(gs[:2, 2])
    ax4 = fig.add_subplot(gs[:2, 4])
    ax5 = fig.add_subplot(gs[:2, 5])
    ax6 = fig.add_subplot(gs[2:4, 1])
    ax7 = fig.add_subplot(gs[2:4, 2])
    ax8 = fig.add_subplot(gs[2:4, 4])
    ax9 = fig.add_subplot(gs[2:4, 5])
    ax10 = fig.add_subplot(gs[2:4, 7])
    ax11 = fig.add_subplot(gs[2:4, 8])
    ax12 = fig.add_subplot(gs[2:4, 10])
    ax13 = fig.add_subplot(gs[2:4, 11])

    ax2.scatter(x_outline, y_outline, s=1, c='black')
    ax2.axis('off')

    ax3.scatter(x_outline, y_outline, s=1, c='black')
    ax3.axis('off')

    ax4.scatter(x_outline, y_outline, s=1, c='black')
    ax4.axis('off')

    ax5.scatter(x_outline, y_outline, s=1, c='black')
    ax5.axis('off')

    ax6.scatter(x_outline, y_outline, s=1, c='black')
    ax6.axis('off')

    ax7.scatter(x_outline, y_outline, s=1, c='black')
    ax7.axis('off')

    ax8.scatter(x_outline, y_outline, s=1, c='black')
    ax8.axis('off')

    ax9.scatter(x_outline, y_outline, s=1, c='black')
    ax9.axis('off')

    ax10.scatter(x_outline, y_outline, s=1, c='black')
    ax10.axis('off')

    ax11.scatter(x_outline, y_outline, s=1, c='black')
    ax11.axis('off')

    ax12.scatter(x_outline, y_outline, s=1, c='black')
    ax12.axis('off')

    ax13.scatter(x_outline, y_outline, s=1, c='black')
    ax13.axis('off')


    ####

    selected_trials = ['trial09', 'trial07', 'trial08', 'trial19']
    participants = ['PPT_004']
    all_CoPs = {}

    i=0
    for trial in selected_trials:
        all_CoPs[trial] = {}

        for foot in feet:


            trial_CoP = np.zeros((0, 2))

            for participant in participants:
                # for participant in participants:

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

                        loc, x, y = centre_of_contact(D, threshold=250)

                        colors = mcp.gen_color_normalized(cmap='winter', data_arr=list(range(100)))


                        trial_CoP = np.vstack((trial_CoP, loc))

            all_CoPs[trial][foot] = trial_CoP

        i += 1


    for i in range(int(len(all_CoPs['trial09']['left'][:,1])/100)-2):

        ax6.scatter(all_CoPs['trial09']['left'][(100*i):((100*i)+100),0], all_CoPs['trial09']['left'][(100*i):((100*i)+100),1], c=colors, s=2.)

        ax7.scatter(all_CoPs['trial09']['right'][(100*i):((100*i)+100),0], all_CoPs['trial09']['right'][(100*i):((100*i)+100),1], c=colors, s=2.)
    ax7.invert_xaxis()


    for i in range(int(len(all_CoPs['trial07']['left'][:,1])/100)-2):

        ax8.scatter(all_CoPs['trial07']['left'][(100*i):((100*i)+100),0], all_CoPs['trial07']['left'][(100*i):((100*i)+100),1], c=colors, s=2.)

        ax9.scatter(all_CoPs['trial07']['right'][(100*i):((100*i)+100),0], all_CoPs['trial07']['right'][(100*i):((100*i)+100),1], c=colors, s=2.)
    ax9.invert_xaxis()


    for i in range(int(len(all_CoPs['trial08']['left'][:,1])/100)-2):

        ax10.scatter(all_CoPs['trial08']['left'][(100*i):((100*i)+100),0], all_CoPs['trial08']['left'][(100*i):((100*i)+100),1], c=colors, s=2.)

        ax11.scatter(all_CoPs['trial08']['right'][(100*i):((100*i)+100),0], all_CoPs['trial08']['right'][(100*i):((100*i)+100),1], c=colors, s=2.)
    ax11.invert_xaxis()



    for i in range(int(len(all_CoPs['trial19']['left'][:,1])/100)-2):

        ax12.scatter(all_CoPs['trial19']['left'][(100*i):((100*i)+100),0], all_CoPs['trial19']['left'][(100*i):((100*i)+100),1], c=colors, s=2.)

        ax13.scatter(all_CoPs['trial19']['right'][(100*i):((100*i)+100),0], all_CoPs['trial19']['right'][(100*i):((100*i)+100),1], c=colors, s=2.)
    ax13.invert_xaxis()

    selected_trials = ['trial05', 'trial06']
    participants = ['PPT_004']
    all_CoPs = {}

    filepath_prefix = '../preprocessed_data/'

    for trial in selected_trials:
        all_CoPs[trial] = {}

        for foot in feet:
            all_CoPs[trial][foot] = {}

            trial_CoP = np.zeros((0, 2))

            for participant in participants:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    # load in the data for the participant
                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    D = data['Raw data']

                    loc, x, y = centre_of_contact(D, threshold=250)

                    colors = mcp.gen_color_normalized(cmap='winter', data_arr=list(range(D.shape[2])))

                    trial_CoP = np.vstack((trial_CoP, loc))


                all_CoPs[trial][foot]['data'] = trial_CoP
                all_CoPs[trial][foot]['colours'] = colors


    ax2.scatter(all_CoPs['trial05']['left']['data'][:,0],
                all_CoPs['trial05']['left']['data'][:,1], c=all_CoPs['trial05']['left']['colours'], s=2.)

    ax3.scatter(all_CoPs['trial05']['right']['data'][:,0],
                all_CoPs['trial05']['right']['data'][:,1], c=all_CoPs['trial05']['right']['colours'], s=2.)
    ax3.invert_xaxis()

    ax4.scatter(all_CoPs['trial06']['left']['data'][:,0],
                all_CoPs['trial06']['left']['data'][:,1], c=all_CoPs['trial06']['left']['colours'], s=2.)

    ax5.scatter(all_CoPs['trial06']['right']['data'][:,0],
                all_CoPs['trial06']['right']['data'][:,1], c=all_CoPs['trial06']['right']['colours'], s=2.)
    ax5.invert_xaxis()

    plt.savefig('../individual_figures/CoC_on_foot.png', dpi=100)


def max_per_trial_plot(all_metrics, step_all_metrics):
    """ Figure showing the maximum force experienced by either foot by each participant across all tasks

    Args:
        all_metrics: dataframe containing analysed data for all tasks
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:

    """
    step_all_metrics = step_all_metrics.drop(columns='Step number')

    normal = step_all_metrics[step_all_metrics['Trials'] == 'trial07']
    slow = step_all_metrics[step_all_metrics['Trials'] == 'trial09']
    fast = step_all_metrics[step_all_metrics['Trials'] == 'trial08']
    jogging = step_all_metrics[step_all_metrics['Trials'] == 'trial15']
    gravel = step_all_metrics[step_all_metrics['Trials'] == 'trial19']
    slope = step_all_metrics[step_all_metrics['Trials'] == 'trial13']
    stairs = step_all_metrics[step_all_metrics['Trials'] == 'trial14']

    standing = all_metrics[all_metrics['Trials'] == 'trial05']
    wobble = all_metrics[all_metrics['Trials'] == 'trial06']
    sit_to_stand = all_metrics[all_metrics['Trials'] == 'trial10']
    sit_to_walk = all_metrics[all_metrics['Trials'] == 'trial11']
    twisting = all_metrics[all_metrics['Trials'] == 'trial12']

    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping']

    df = pd.concat([normal, slow, fast, jogging, slope, stairs, gravel, standing, wobble, twisting, jumping, sit_to_stand, sit_to_walk])

    sorted_trial_titles = ['Quiet standing', 'Wobble-board', 'Twisting','Sit-to-stand', 'Sit-to-walk','Slow walking', \
                           'Normal walking', 'Fast walking', 'Jogging', 'Slope walking','Stairs','Gravel walking',\
                           'Left foot jump', 'Right foot jump', 'Both feet jump']

    max_vals_df = pd.DataFrame({'Trials': [], 'Participant': [],
                                'Maximum force': []})

    for trial in sorted_trial_titles:

        trial_df = df[df['Trial title'] == trial]

        participants = pd.unique(trial_df['Participant'])

        for participant in participants:

            ppt_df = trial_df[trial_df['Participant'] == participant]

            maximum = ppt_df['Force'].max()

            di = {'Trials': trial, 'Participant': participant,
                  'Maximum force': pd.Series(maximum)}
            max_vals = pd.DataFrame(di)

            max_vals_df = pd.concat([max_vals_df, max_vals])

    sorted_colours = ['indigo', 'purple', 'mediumorchid','rebeccapurple','blueviolet', 'mediumaquamarine', 'limegreen', 'lime', 'forestgreen', 'mediumseagreen','chartreuse', 'greenyellow',
               'crimson', 'indianred', 'red']

    sns.set_palette(sorted_colours)
    plt.figure(figsize=(10, 10), dpi=100)
    ax = sns.swarmplot(data=max_vals_df, x='Trials', y='Maximum force',
                       order=['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk',
                              'Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking', 'Stairs', \
                              'Gravel walking', 'Left foot jump', 'Right foot jump', 'Both feet jump'],
                       hue='Trials',
                       hue_order=['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk', 'Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking','Stairs',\
                              'Gravel walking', 'Left foot jump', 'Right foot jump', 'Both feet jump'],
                       size=6.5)
    sns.boxplot(data=max_vals_df, x='Trials', y='Maximum force', ax=ax, color='white', linewidth=2.5)
    plt.xticks(rotation=80, fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend([], [], frameon=False)
    sns.despine(ax=ax)
    ax.set_ylabel('Force \n Percent body mass (%)', fontsize=20)
    ax.set_xlabel('Task', fontsize=20)
    plt.subplots_adjust(bottom=0.5)
    plt.savefig("../individual_figures/max per trial.png")

def mean_per_trial_plot(step_all_metrics, all_metrics):
    """ Figure showing the mean force experienced by either foot by each participant across all tasks

    Args:
        all_metrics: dataframe containing analysed data for all tasks
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:

    """

    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]
    all_metrics = all_metrics[all_metrics['Force'] > 5.]

    normal = step_all_metrics[step_all_metrics['Trials'] == 'trial07']

    slow = step_all_metrics[step_all_metrics['Trials'] == 'trial09']
    fast = step_all_metrics[step_all_metrics['Trials'] == 'trial08']
    jogging = step_all_metrics[step_all_metrics['Trials'] == 'trial15']
    gravel = step_all_metrics[step_all_metrics['Trials'] == 'trial19']
    slope = step_all_metrics[step_all_metrics['Trials'] == 'trial13']
    stairs = step_all_metrics[step_all_metrics['Trials'] == 'trial14']

    standing = all_metrics[all_metrics['Trials'] == 'trial05']
    wobble = all_metrics[all_metrics['Trials'] == 'trial06']
    sit_to_stand = all_metrics[all_metrics['Trials'] == 'trial10']
    sit_to_walk = all_metrics[all_metrics['Trials'] == 'trial11']
    twisting = all_metrics[all_metrics['Trials'] == 'trial12']


    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping']

    df = pd.concat([normal, slow, fast, jogging, slope, stairs, gravel, standing, wobble, twisting, jumping, sit_to_stand, sit_to_walk])

    sorted_trial_titles = ['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk', 'Slow walking', \
                           'Normal walking', 'Fast walking', 'Jogging', 'Slope walking', 'Stairs', 'Gravel walking', \
                           'Left foot jump', 'Right foot jump', 'Both feet jump']

    mean_vals_df = pd.DataFrame({'Trials': [], 'Participant': [],
                                'Mean force': []})

    for trial in sorted_trial_titles:

        trial_df = df[df['Trial title'] == trial]

        participants = pd.unique(trial_df['Participant'])

        for participant in participants:

            ppt_df = trial_df[trial_df['Participant'] == participant]

            mean = ppt_df['Force'].mean()

            di = {'Trials': trial, 'Participant': participant,
                  'Mean force': pd.Series(mean)}
            mean_vals = pd.DataFrame(di)

            mean_vals_df = pd.concat([mean_vals_df, mean_vals])

    sorted_colours = ['indigo', 'purple', 'mediumorchid', 'rebeccapurple', 'blueviolet', 'mediumaquamarine',
                      'limegreen', 'lime', 'forestgreen', 'mediumseagreen', 'chartreuse', 'greenyellow',
                      'crimson', 'indianred', 'red']

    sns.set_palette(sorted_colours)
    plt.figure(figsize=(10, 10), dpi=100)
    ax = sns.swarmplot(data=mean_vals_df, x='Trials', y='Mean force',
                       order=['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk',
                              'Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking', 'Stairs', \
                              'Gravel walking', 'Left foot jump', 'Right foot jump', 'Both feet jump'],
                       hue='Trials',
                       hue_order=['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk', 'Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking','Stairs',\
                              'Gravel walking', 'Left foot jump', 'Right foot jump', 'Both feet jump'],
                       size=6.5)
    sns.boxplot(data=mean_vals_df, x='Trials', y='Mean force', ax=ax, color='white', linewidth=2.5)
    plt.xticks(rotation=80, fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend([], [], frameon=False)
    sns.despine(ax=ax)
    ax.set_ylabel('Force \n Percent body mass (%)', fontsize=20)
    ax.set_xlabel('Task', fontsize=20)
    plt.subplots_adjust(bottom=0.5)
    plt.savefig("../individual_figures/mean per trial.png")

def mean_per_trial_plot_area(step_all_metrics, all_metrics):
    """ Figure showing the mean contact area experienced by either foot by each participant across all tasks

    Args:
        all_metrics: dataframe containing analysed data for all tasks
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:

    """

    step_all_metrics = step_all_metrics.drop(columns='Step number')

    normal = step_all_metrics[step_all_metrics['Trials'] == 'trial07']

    slow = step_all_metrics[step_all_metrics['Trials'] == 'trial09']
    fast = step_all_metrics[step_all_metrics['Trials'] == 'trial08']
    jogging = step_all_metrics[step_all_metrics['Trials'] == 'trial15']
    gravel = step_all_metrics[step_all_metrics['Trials'] == 'trial19']
    slope = step_all_metrics[step_all_metrics['Trials'] == 'trial13']
    stairs = step_all_metrics[step_all_metrics['Trials'] == 'trial14']

    standing = all_metrics[all_metrics['Trials'] == 'trial05']
    wobble = all_metrics[all_metrics['Trials'] == 'trial06']
    sit_to_stand = all_metrics[all_metrics['Trials'] == 'trial10']
    sit_to_walk = all_metrics[all_metrics['Trials'] == 'trial11']
    twisting = all_metrics[all_metrics['Trials'] == 'trial12']


    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping']

    df = pd.concat([normal, slow, fast, jogging, slope, stairs, gravel, standing, wobble, twisting, jumping, sit_to_stand, sit_to_walk])

    sorted_trial_titles = ['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk', 'Slow walking', \
                           'Normal walking', 'Fast walking', 'Jogging', 'Slope walking', 'Stairs', 'Gravel walking', \
                           'Left foot jump', 'Right foot jump', 'Both feet jump']

    mean_vals_df = pd.DataFrame({'Trials': [], 'Participant': [],
                                'Mean contact area': []})

    for trial in sorted_trial_titles:

        trial_df = df[df['Trial title'] == trial]

        participants = pd.unique(trial_df['Participant'])

        for participant in participants:

            ppt_df = trial_df[trial_df['Participant'] == participant]

            mean = ppt_df['Contact area percent'].mean()


            di = {'Trials': trial, 'Participant': participant,
                  'Mean contact area': pd.Series(mean)}
            mean_vals = pd.DataFrame(di)

            mean_vals_df = pd.concat([mean_vals_df, mean_vals])

    sorted_colours = ['indigo', 'purple', 'mediumorchid','rebeccapurple','blueviolet', 'mediumaquamarine', 'limegreen', 'lime', 'forestgreen', 'mediumseagreen','chartreuse', 'greenyellow',
               'crimson', 'indianred', 'red']


    sns.set_palette(sorted_colours)
    plt.figure(figsize=(10, 10), dpi=100)
    ax = sns.swarmplot(data=mean_vals_df, x='Trials', y='Mean contact area',
                       order=['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk', 'Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking','Stairs',\
                              'Gravel walking', 'Left foot jump', 'Right foot jump', 'Both feet jump'],
                       hue='Trials',
                       hue_order=['Quiet standing', 'Wobble-board', 'Twisting', 'Sit-to-stand', 'Sit-to-walk', 'Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking','Stairs',\
                              'Gravel walking', 'Left foot jump', 'Right foot jump', 'Both feet jump'],
                       size=6.5)
    sns.boxplot(data=mean_vals_df, x='Trials', y='Mean contact area', ax=ax, color='white', linewidth=2.5)
    plt.xticks(rotation=80, fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend([], [], frameon=False)
    sns.despine(ax=ax)
    ax.set_ylabel('Contact area (%)', fontsize=20)
    ax.set_xlabel('Task', fontsize=20)
    plt.subplots_adjust(bottom=0.5)
    plt.savefig("../individual_figures/mean per trial area.png")

def pressure_area_correlation_plots(step_all_metrics, all_metrics):
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

    plt.figure(figsize=(10,10), dpi=100)
    sns.jointplot(data=minimal, x='Contact area percent', y='Force', kind='hex', color='darkviolet',
                  height=10, ratio=2)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xticks(fontsize=25)
    plt.ylabel('Force', fontsize=30)
    plt.xlabel('Contact area (%)', fontsize=30)
    plt.savefig('../individual_figures/minimal_pressure_area_correlation.png', dpi=100,
                bbox_inches="tight")

    min_shapiro_area = stats.shapiro(minimal['Contact area percent'])
    min_shapiro_pres = stats.shapiro(minimal['Force'])
    min_corr = stats.spearmanr(minimal['Contact area percent'], minimal['Force'])

    print('area: ', min_shapiro_area)
    print('pressure: ', min_shapiro_pres)
    print('R2: ', min_corr[0]**2)
    print('df: ', minimal['Contact area percent'].shape[0] - 2)

    step_all_metrics = step_all_metrics.drop(columns='Step number')
    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]

    plt.figure(figsize=(10, 10), dpi=100)
    sns.jointplot(data=step_all_metrics, x='Contact area percent', y='Force', kind='hex',
                  color='springgreen')
    plt.xlim(0, 100)
    plt.ylim(0, 200)
    plt.xticks(fontsize=25)
    plt.ylabel('Force', fontsize=30)
    plt.xlabel('Contact area (%)', fontsize=30)
    plt.savefig('../individual_figures/walking_pressure_area_correlation.png', dpi=100,
                bbox_inches="tight")

    step_shapiro_area = stats.shapiro(step_all_metrics['Contact area percent'])
    step_shapiro_pres = stats.shapiro(step_all_metrics['Force'])
    step_corr = stats.spearmanr(step_all_metrics['Contact area percent'], step_all_metrics['Force'])

    print('area: ', step_shapiro_area)
    print('pressure: ', step_shapiro_pres)
    print('R2: ', step_corr[0]**2)
    print('df: ', step_all_metrics['Contact area percent'].shape[0] - 2)

    plt.figure(figsize=(10, 10), dpi=100)
    sns.jointplot(data=jumping, x='Contact area percent', y='Force', kind='hex', color='red',
                  height=10, ratio=2)
    plt.xlim(0, 100)
    plt.ylim(0, 250)
    plt.xticks(fontsize=25)
    plt.ylabel('Force', fontsize=30)
    plt.xlabel('Contact area (%)', fontsize=30)
    plt.savefig(
        '/Users/lukecleland/Documents/PhD/Research projects/Project insole/individual_figures/panels/jumping_pressure_area_correlation.png',
        dpi=100,
        bbox_inches="tight")

    jump_shapiro_area = stats.shapiro(jumping['Contact area percent'])
    jump_shapiro_pres = stats.shapiro(jumping['Force'])
    jump_corr = stats.spearmanr(jumping['Contact area percent'], jumping['Force'])

    print('area: ', jump_shapiro_area)
    print('pressure: ', jump_shapiro_pres)
    print('R2: ', jump_corr[0] ** 2)
    print('df: ', jumping['Contact area percent'].shape[0] - 2)

def CoP_density(all_CoP_coordinates, step_CoP_coordinates):
    """ Plot 2D histograms of CoP location to show density

    Args:
        all_CoP_coordinates: dataframe containing CoP locations at all timepoints for all tasks
        step_CoP_coordinates: dataframe containing CoP locations at all timepoints for walking trials calculated using
            normalized steps

    Returns:

    """
    x_outline, y_outline = get_foot_outline()

    all_CoP_coordinates = all_CoP_coordinates[all_CoP_coordinates['Trial type'] != 'Locomotion']
    CoPs = pd.concat([step_CoP_coordinates, all_CoP_coordinates])

    sorted_trials = ['trial05', 'trial06', 'trial12', 'trial10', 'trial11', \
                     'trial09', 'trial07', 'trial08', 'trial15', 'trial13', 'trial14', 'trial19', \
                     'trial16', 'trial17', 'trial18']

    fig = plt.figure(constrained_layout=True, figsize=(15, 4), dpi=100)
    plt.rcParams.update({'font.size': 5})
    gs = GridSpec(1, 15, figure=fig)

    for i in range(len(sorted_trials)):
        ax = fig.add_subplot(gs[i])
        trial_df = CoPs[CoPs['Trial'] == sorted_trials[i]]
        sns.histplot(data=trial_df, x='X_pressure', y='Y_pressure', \
                     cbar=False, ax=ax, \
                     cmap='rainbow', cbar_kws={'label': 'probability'}, binwidth=[5, 10], stat='probability')
        ax.scatter(x_outline, y_outline, c='black', s=2)
        ax.set_title(trial_titles[sorted_trials[i]])
        ax.axis('off')

    plt.savefig("../individual_figures/CoP_density_plot.png", dpi=100, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(2, 10), dpi=100)
    plt.rcParams.update({'font.size': 50})

    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical', ticks=[0, 1])
    cb1.ax.set_yticklabels(['0', '1'])  # horizontal colorbar
    cb1.set_label('Probability', fontsize=50)

    plt.savefig("../individual_figures/CoP_density_colorbar.png", dpi=100, bbox_inches="tight")

def CoP_trace_colorbar():
    """ Colorbar to show trial progress and step progress for CoP traces

    Returns:
            colorbar

    """
    fig, ax = plt.subplots(figsize=(2, 10), dpi=100)
    plt.rcParams.update({'font.size': 50})

    cmap = mpl.cm.spring
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical', ticks=[0, 1])
    cb1.ax.set_yticklabels(['0', '100'])  # horizontal colorbar
    cb1.set_label('Trial progress (%)', fontsize=50)

    plt.savefig("../individual_figures/CoP_cbar_standing.png", dpi=100, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(2, 10), dpi=100)
    plt.rcParams.update({'font.size': 50})

    cmap = mpl.cm.spring
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical', ticks=[0, 1])
    cb1.ax.set_yticklabels(['0', '100'])  # horizontal colorbar
    cb1.set_label('Step cycle (%)', fontsize=50)

    plt.savefig("../individual_figures/CoP_cbar_walking.png", dpi=100, bbox_inches="tight")


def CoC_trace_colorbar():
    """ Colorbar to show trial progress and step progress for CoC traces

    Returns:
            colorbar

    """
    fig, ax = plt.subplots(figsize=(2, 10), dpi=100)
    plt.rcParams.update({'font.size': 50})

    cmap = mpl.cm.winter
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical', ticks=[0, 1])
    cb1.ax.set_yticklabels(['0', '100'])  # horizontal colorbar
    cb1.set_label('Trial progress (%)', fontsize=50)

    plt.savefig("../individual_figures/CoC_cbar_standing.png", dpi=100, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(2, 10), dpi=100)
    plt.rcParams.update({'font.size': 50})

    cmap = mpl.cm.winter
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical', ticks=[0, 1])
    cb1.ax.set_yticklabels(['0', '100'])  # horizontal colorbar
    cb1.set_label('Step cycle (%)', fontsize=50)

    plt.savefig("../individual_figures/CoC_cbar_walking.png", dpi=100, bbox_inches="tight")

def CoP_stacked_bars(all_metrics, step_all_metrics):
    """ Stacked bar chart to show proportion of time CoP is in each of the 4 regions of the foot

    Args:
        all_metrics: dataframe containing analysed data for all tasks
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:

    """
    step_all_metrics = step_all_metrics.drop(columns='Step number')

    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]
    step_all_metrics[step_all_metrics['CoP location'] != 'NA']

    count_walk = (step_all_metrics['CoP location'].value_counts() / step_all_metrics['CoP location'].count()) * 100
    count_walk_df = pd.DataFrame(count_walk).rename(columns={'CoP location': 'Locomotion'})


    all_metrics = all_metrics[all_metrics['Force'] > 5.]
    all_metrics[all_metrics['CoP location'] != 'NA']
    all_metrics_no_jumping = all_metrics[all_metrics['Trial type'] != 'Jumping']
    minimal = all_metrics_no_jumping[all_metrics_no_jumping['Trial type'] == 'Standing']

    count_min = (minimal['CoP location'].value_counts() / minimal['CoP location'].count()) * 100
    count_min_df = pd.DataFrame(count_min).rename(columns={'CoP location': 'Standing'})

    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping']

    count_jump = (jumping['CoP location'].value_counts() / jumping['CoP location'].count()) * 100
    count_jump_df = pd.DataFrame(count_jump).rename(columns={'CoP location': 'Jumping'})

    #df = all_metrics[all_metrics['Force'] > 5.]
    #df = df[df['CoP location'] != 'NA']
    #df['Change in force'] = df['Change in force'].abs()

    #change = df.nlargest(round(df.shape[0] * .01), 'Change in force', keep='all')

    #count_change = (change['CoP location'].value_counts() / change['CoP location'].count()) * 100
    #count_change_df = pd.DataFrame(count_change).rename(columns={'CoP location': 'Greatest change'})

    #count_df = pd.concat([count_walk_df, count_min_df, count_change_df], axis='columns')
    count_df = pd.concat([count_min_df, count_walk_df, count_jump_df], axis='columns')

    regions = ['H', 'A', 'M', 'T']
    count_df = count_df.loc[regions]

    print(count_df)

    plt.figure(figsize=(10,80), dpi=100)
    count_df.T.plot(kind='bar', stacked=True, color=['crimson', 'darkorchid', 'lightseagreen', 'limegreen'], legend=False)
    plt.ylabel('Percentage of time', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    sns.despine()
    plt.savefig("../individual_figures/CoP_stacked_bar.png", dpi=100, bbox_inches="tight")

def pressure_area_combinations(all_metrics, step_all_metrics):
    """ Figure showing mean and SD of pressure and contact area for each of the 3 task types

    Args:
        all_metrics: dataframe containing analysed data for all tasks
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:

    """
    step_all_metrics = step_all_metrics.drop(columns='Step number')

    all_metrics = all_metrics[all_metrics['Force'] > 5.]
    all_metrics[all_metrics['CoP location'] != 'NA']
    all_metrics_no_jumping = all_metrics[all_metrics['Trial type'] != 'Jumping']

    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]
    step_all_metrics[step_all_metrics['CoP location'] != 'NA']

    minimal = all_metrics_no_jumping[all_metrics_no_jumping['Trial type'] == 'Standing']

    new_means = minimal.groupby('CoP location').mean().rename(columns={'Force': 'Mean pressure',
                                                                       'Contact area percent': 'Mean area',
                                                                       'Change in force': 'Mean change'})
    new_stds = minimal.groupby('CoP location').std().rename(columns={'Force': 'Std pressure',
                                                                     'Contact area percent': 'Std area',
                                                                     'Change in force': 'Std change'})

    new_combined = pd.concat([new_means, new_stds], axis=1)
    regions = ['H', 'A', 'M', 'T']
    new_combined = new_combined.loc[regions]
    color = ['crimson', 'darkorchid', 'lightseagreen', 'limegreen']


    plt.figure(figsize=(10,10), dpi=100)
    plt.title('Standing', fontsize=60)
    i = 0
    for row in new_combined.iterrows():
        plt.errorbar(x=row[1]['Mean area'], y=row[1]['Mean pressure'], \
                     xerr=row[1]['Std area'], yerr=row[1]['Std pressure'], color=color[i], markersize=8,
                     marker='o', capthick=2, capsize=5, elinewidth=2)
        i += 1
    sns.despine()
    plt.xlabel('Contact area percent', fontsize=50)
    plt.xticks(fontsize=45)
    plt.ylabel('Force', fontsize=50)
    plt.yticks(fontsize=45)
    plt.xlim(0, 100)
    plt.ylim(0, 75)
    plt.savefig("../individual_figures/minimal_pressure_area_combinations.png", dpi=100)



    new_means_walk = step_all_metrics.groupby('CoP location').mean().rename(columns={'Force': 'Mean pressure',
                                                                       'Contact area percent': 'Mean area',
                                                                       'Change in force': 'Mean change'})
    new_stds_walk = step_all_metrics.groupby('CoP location').std().rename(columns={'Force': 'Std pressure',
                                                                     'Contact area percent': 'Std area',
                                                                     'Change in force': 'Std change'})

    new_combined_walk = pd.concat([new_means_walk, new_stds_walk], axis=1)
    regions = ['H', 'A', 'M', 'T']
    new_combined_walk = new_combined_walk.loc[regions]
    color = ['crimson', 'darkorchid', 'lightseagreen', 'limegreen']

    plt.figure(figsize=(10, 10), dpi=100)
    plt.title('Locomotion', fontsize=60)
    i = 0
    for row in new_combined_walk.iterrows():
        plt.errorbar(x=row[1]['Mean area'], y=row[1]['Mean pressure'], \
                     xerr=row[1]['Std area'], yerr=row[1]['Std pressure'], color=color[i], markersize=8,
                     marker='o', capthick=2, capsize=5, elinewidth=2)
        i += 1
    sns.despine()
    plt.xlabel('Contact area percent', fontsize=50)
    plt.xticks(fontsize=45)
    plt.ylabel('Force', fontsize=50)
    plt.yticks(fontsize=45)
    plt.xlim(0,100)
    plt.ylim(0,150)
    plt.savefig("../individual_figures/walk_pressure_area_combinations.png", dpi=100)

    # df = all_metrics[all_metrics['Force'] > 5.]
    # df = df[df['CoP location'] != 'NA']
    # df['Change in force'] = df['Change in force'].abs()

    # change = df.nlargest(round(df.shape[0] * .01), 'Change in force', keep='all')
    # change.describe().to_csv('../descriptive_statistic_csvs/change_descibe.csv')
    # change.groupby('CoP location').describe().to_csv('../descriptive_statistic_csvs/change_descibe_location.csv')

    # new_means_change = change.groupby('CoP location').mean().rename(
    #    columns={'Force': 'Mean pressure',
    #             'Contact area percent': 'Mean area',
    #             'Change in force': 'Mean change'})
    # new_stds_change = change.groupby('CoP location').std().rename(columns={'Force': 'Std pressure',
    #                                                                               'Contact area percent': 'Std area',
    #                                                                               'Change in force': 'Std change'})

    # new_combined_change = pd.concat([new_means_change, new_stds_change], axis=1)

    jumping = all_metrics[all_metrics['Trial type'] == 'Jumping']

    new_means_jump = jumping.groupby('CoP location').mean().rename(columns={'Force': 'Mean pressure',
                                                                            'Contact area percent': 'Mean area',
                                                                            'Change in force': 'Mean change'})
    new_stds_jump = jumping.groupby('CoP location').std().rename(columns={'Force': 'Std pressure',
                                                                          'Contact area percent': 'Std area',
                                                                          'Change in force': 'Std change'})

    new_combined_jump = pd.concat([new_means_jump, new_stds_jump], axis=1)
    regions = ['H', 'A', 'M', 'T']
    new_combined_jump = new_combined_jump.loc[regions]

    regions = ['H', 'A', 'M', 'T']
    new_combined_change = new_combined_jump.loc[regions]
    color = ['crimson', 'darkorchid', 'lightseagreen', 'limegreen']

    plt.figure(figsize=(10, 10), dpi=100)
    plt.title('Jumping', fontsize=50)
    i = 0
    for row in new_combined_jump.iterrows():
        plt.errorbar(x=row[1]['Mean area'], y=row[1]['Mean pressure'], \
                     xerr=row[1]['Std area'], yerr=row[1]['Std pressure'], color=color[i], markersize=8,
                     marker='o', capthick=2, capsize=5, elinewidth=2)
        i += 1
    sns.despine()
    plt.xlabel('Contact area percent', fontsize=50)
    plt.xticks(fontsize=45)
    plt.ylabel('Force', fontsize=50)
    plt.yticks(fontsize=45)
    plt.xlim(0, 100)
    plt.ylim(0, 300)
    plt.savefig("../individual_figures/jump_pressure_area_combinations.png", dpi=100)

def sort_nmf_df(dfs):
    """ Sorts dataframes generated from run_nmf to order components in descending order of variance explained

    Args:
        dfs (dict): dictionary contianing dataframes generated from NMF analysis in run_nmf

    Returns:
        df (dict): dictionary containing sorted dataframes

    """
    for trial in dfs:

        total_var_explained = np.asarray(dfs[trial]['Variance explained'])
        diff = np.zeros(len(total_var_explained))

        total_var_explained = np.append(np.zeros(1), total_var_explained)
        for i in range(dfs[trial].shape[0] - 1):
            diff[i] = total_var_explained[i + 1] - total_var_explained[i]

        diff = np.abs(diff)

        dfs[trial]['Difference'] = np.abs(diff)

        dfs[trial] = dfs[trial].sort_values(by='Difference', ascending=False)

        tot = np.cumsum(np.asarray(dfs[trial]['Difference']))

        dfs[trial]['Cumulative variance explained'] = tot
        dfs[trial]['Order'] = list(range(30))

    return dfs

def nmf_scree_plot(dfs, **args):
    """ Scree plot of cumulative variance explained (y axis) for each component (x axis) for all tasks

    Args:
        dfs (dict): dictionary containing sorted dataframes
        **args:

    Returns:
        Scree plot
    """

    #rials = args.get('trials', 'all')
    #if trials == 'all':
    #    trial_ids = trial_ids[5:-1]
    #else:
    #    trial_ids = trials

    trial_titles = ['Quiet standing', 'Wobble-board', 'Twisting', 'Sit to stand', 'Sit to walk', 'Slow', 'Normal', 'Fast',
                    'Jogging', 'Slope walking', 'Stairs', \
                    'Gravel', 'Both', 'Left', 'Right']

    plt.figure(figsize=(12, 20), dpi=100)

    trials = trial_ids[5:-1]
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks")
    for i in range(len(trials)):
        sns.lineplot(data=dfs[trials[i]], x='Order', y='Cumulative variance explained', color=trial_colors[trials[i]],
                     label=trial_titles[i], marker='o', markersize=4)
    plt.xlim(0, 31)
    plt.ylim(0, 1)
    plt.ylabel('Cumulative variance explained', fontsize=50)
    plt.xlabel('Component', fontsize=50)
    plt.axhline(y=.8, c='r', lw='4', linestyle='--')
    plt.legend(loc='lower right', fontsize=35)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylim(.3, 1.)
    plt.xlim(0, 15)
    plt.tight_layout()
    sns.despine()
    plt.savefig(
        '../individual_figures/NMF_variance_explained_all.png',
        dpi=100)


def trial_type_nmf_scree_plot(trial_type_df):
    """ Scree plot of cumulative variance explained (y axis) for each component (x axis) for all task types

    Args:
        dfs (dict): dictionary containing sorted dataframes
        **args:

    Returns:
        Scree plot
    """

    plt.figure(figsize=(12, 20), dpi=100)

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks")
    for i in range(len(list(trial_type_df.keys()))):
        sns.lineplot(data=trial_type_df[list(trial_type_df.keys())[i]], x='Order', y='Cumulative variance explained', color=trial_classification_colour_palette[i],
                     label=trial_types[i], marker='o', markersize=4)
    plt.xlim(0, 31)
    plt.ylim(0, 1)
    plt.ylabel('Cumulative variance explained', fontsize=50)
    plt.xlabel('Component', fontsize=50)
    plt.axhline(y=.8, c='r', lw='4', linestyle='--')
    plt.legend(loc='lower right', fontsize=35)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylim(.3, 1.)
    plt.xlim(0, 15)
    plt.tight_layout()
    sns.despine()
    plt.savefig(
        '../individual_figures/NMF_variance_explained_trial_type.png',
        dpi=100)

def nmf_components_required_on_foot(dfs, models, scaled):
    """ Plot first 15 components for each task, and bar chart showing number of components required to explained
        80% variance explained

    Args:
        dfs (dict): dictionary containing sorted dataframes
        models (dict): dictionary containing NMF component and weight matrices
        scaled (dict): dictionary containing all participant data scaled to a uniform size per task

    Returns:

    """

    x_outline, y_outline = get_foot_outline()

    trial_ids = ['trial05', 'trial06', 'trial12', 'trial10', 'trial11', 'trial09', 'trial07', 'trial08', 'trial15',
                 'trial13', 'trial14', 'trial19', 'trial16', 'trial17', 'trial18']

    components_required = {}

    for i in range(len(trial_ids)):
        W = models[trial_ids[i]]['W']
        H = models[trial_ids[i]]['H']

        li = np.asarray(dfs[trial_ids[i]]['Component'])

        array = np.asarray(dfs[trial_ids[i]]['Cumulative variance explained'])
        for j in range(len(array)):
            if array[j] > .8:
                num = j
                break

        components_required[trial_titles[trial_ids[i]]] = num

        W_new = np.zeros((0, W.shape[0]))
        H_new = np.zeros((0, H.shape[1]))

        for k in range(15):
            W_new = np.vstack((W_new, W[:, li[k]]))
            H_new = np.vstack((H_new, H[li[k], :]))

        s = scaled[trial_ids[i]]['Stimulus']

        plt.figure(figsize=(3 * 15, 6))
        for l in range(H_new[:15].shape[0]):
            plt.subplot(1, 15, l + 1)
            plt.title(trial_titles[trial_ids[i]] + ' \n Component ' + str(li[l]))
            plt.scatter(s.location[:, 0], s.location[:, 1],
                        c=H_new[l], cmap='RdPu', marker=",", vmin=0)
            plt.scatter(x_outline, y_outline, s=3., c='black')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(
            '../individual_figures/nmf_components_' +
            trial_titles[trial_ids[i]] + '.png',
            dpi=100)


    sorted_trial_colors = ['indigo', 'purple', 'mediumorchid', 'rebeccapurple', 'blueviolet', 'mediumaquamarine', 'limegreen',
                    'lime', 'forestgreen', 'mediumseagreen', 'chartreuse', 'greenyellow',
                    'crimson', 'indianred', 'red']
    sorted_trial_titles = ['Quiet standing', 'Wobble-board','Twisting','Sit-to-stand','Sit-to-walk',\
                           'Slow walking', 'Normal walking','Fast walking','Jogging','Slope walking','Stairs','Gravel walking',\
                           'Both feet jump','Left foot jump','Right foot jump']
    com_req_new = {}
    for trial in sorted_trial_titles:
        com_req_new[trial] = components_required[trial]

    plt.figure(figsize=(20, 16), dpi=100)
    plt.title('Components required to expalain 80% variance', fontsize=50)
    plt.bar(range(len(com_req_new)), list(com_req_new.values()), align='center', color=sorted_trial_colors)
    plt.xticks(range(len(com_req_new)), list(com_req_new.keys()))
    plt.xticks(rotation=80, fontsize=35)
    plt.yticks(fontsize=35)
    plt.xlabel('Task', fontsize=40)
    plt.ylabel('Number of components', fontsize=40)
    plt.ylim(0, 10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        '../individual_figures/components_required.png',
        dpi=100)

def nmf_walking_hotspots(dfs, models, scaled):
    """ Figure showing the areas of highest weight for each component for walking tasks

    Args:
        dfs (dict): dictionary containing sorted dataframes
        models (dict): dictionary containing NMF component and weight matrices
        scaled (dict): dictionary containing all participant data scaled to a uniform size per task

    Returns:

    """
    trial_ids = ['trial09', 'trial07', 'trial08', 'trial15', 'trial19']
    selected_colors = ['lime', 'cyan', 'magenta', 'red', 'yellow']
    trial_titles = ['Slow', 'Normal', 'Fast', 'Jogging', 'Gravel']

    x_outline, y_outline = get_foot_outline()

    plt.figure(figsize=(8, 3), dpi=100)
    for j in range(3):

        plt.subplot(1, 5, j + 1)
        plt.title('Component ' + str(j + 1))
        plt.scatter(x_outline, y_outline, s=3., c='black')

        for i in range(len(trial_ids)):
            W = models[trial_ids[i]]['W']
            H = models[trial_ids[i]]['H']

            li = np.asarray(dfs[trial_ids[i]]['Component'])

            array = np.asarray(dfs[trial_ids[i]]['Cumulative variance explained'])
            for l in range(len(array)):
                if array[l] > .8:
                    num = l
                    break

            W_new = np.zeros((0, W.shape[0]))
            H_new = np.zeros((0, H.shape[1]))

            for k in range(num):
                W_new = np.vstack((W_new, W[:, li[k]]))
                H_new = np.vstack((H_new, H[li[k], :]))

            s = scaled[trial_ids[i]]['Stimulus']

            ind = np.argpartition(H_new[j], - int(len(H_new[j]) * .025))[-int(len(H_new[j]) * .025):]

            x_ = s.location[ind, 0]
            y_ = s.location[ind, 1]

            points = np.vstack((x_, y_)).T

            hull = ConvexHull(points)

            plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'k', alpha=0.3, c=selected_colors[i],
                     label=trial_titles[i])

            if j == 0:
                plt.legend(bbox_to_anchor=(-.05, 1))

        plt.axis('off')

    plt.tight_layout()
    plt.savefig(
        '../individual_figures/NMF_walking_trials.png',
        dpi=100)

def walking_trial_CoPs(normalized_steps):
    """ Generates dataframes containing the CoP coordinates along the x and y axis for all walking tasks

    Args:
        normalized_steps (dict): dictionary containing pressure data for each step in each walking task
            for each participant

    Returns:
        Dataframe containign CoP coordinates
    """

    all_CoPs = {}

    for trial in normalized_steps:
        all_CoPs[trial] = {}

        for foot in feet:
            all_CoPs[trial][foot] = {}

            for participant in normalized_steps[trial]:
                # for participant in participants:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    all_CoPs[trial][foot][participant] = np.zeros((0, 2))

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

                        loc, x, y = centre_of_pressure(D, threshold=100)

                        remove = []
                        for j in range(loc.shape[0]):
                            if np.isnan(loc[j]).any():
                                remove.append(j)

                        loc = np.delete(loc, remove, axis=0)

                    all_CoPs[trial][foot][participant] = np.vstack((all_CoPs[trial][foot][participant], loc))


    return all_CoPs

def standing_trial_CoPs(stomps):
    """ Generates dataframes containing the CoP coordinates along the x and y axis for all standing tasks

    Args:
        stomps (dict): timepoints at which the stomp occurred at the start and end of the trial

    Returns:
        Dataframe containign CoP coordinates
    """

    trial_ids = ['trial05', 'trial06', 'trial12', 'trial10', 'trial11']

    standing_CoPs = {}

    filepath_prefix = '../preprocessed_data/'

    for trial in trial_ids:
        standing_CoPs[trial] = {}

        for foot in feet:
            standing_CoPs[trial][foot] = {}

            for participant in participant_ids:
                # for participant in participants:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    stomp_start = stomps[trial][participant][0] + 5
                    stomp_end = stomps[trial][participant][1] - 5

                    standing_CoPs[trial][foot][participant] = np.zeros((0, 2))

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    if trial == 'trial05':
                        loc = data['CoP coordinates'][1000:5500]
                    elif trial == 'trial06':
                        loc = data['CoP coordinates'][1000:4000]
                    elif trial == 'trial10':
                        stand_end = trial_10_segmentation_times[participant][0]
                        loc = data['CoP coordinates'][stomp_start:stand_end]
                    elif trial == 'trial11':
                        stand_end = trial_11_segmentation_times[participant][0]
                        loc = data['CoP coordinates'][stomp_start:stand_end]
                    else:
                        loc = data['CoP coordinates'][stomp_start:stomp_end]

                    remove = []
                    for j in range(loc.shape[0]):
                        if np.isnan(loc[j]).any():
                            remove.append(j)

                    loc = np.delete(loc, remove, axis=0)

                    standing_CoPs[trial][foot][participant] = loc

    return standing_CoPs

def jumping_trial_CoPs(stomps):
    """ Generates dataframes containing the CoP coordinates along the x and y axis for all jumping tasks

    Args:
        stomps (dict): timepoints at which the stomp occurred at the start and end of the trial

    Returns:
        Dataframe containign CoP coordinates
    """

    trial_ids = ['trial16','trial17','trial18']

    jumping_CoPs = {}

    filepath_prefix = '../preprocessed_data/'

    for trial in trial_ids:
        jumping_CoPs[trial] = {}

        for foot in feet:
            jumping_CoPs[trial][foot] = {}

            for participant in participant_ids:
                # for participant in participants:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    stomp_start = stomps[trial][participant][0] + 5
                    stomp_end = stomps[trial][participant][1] - 5

                    jumping_CoPs[trial][foot][participant] = np.zeros((0, 2))

                    file = filepath_prefix + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
                    data = decompress_pickle(file)

                    if trial == 'trial05':
                        loc = data['CoP coordinates'][1000:5500]
                    elif trial == 'trial06':
                        loc = data['CoP coordinates'][1000:4000]
                    elif trial == 'trial10':
                        stand_end = trial_10_segmentation_times[participant][0]
                        loc = data['CoP coordinates'][stomp_start:stand_end]
                    elif trial == 'trial11':
                        stand_end = trial_11_segmentation_times[participant][0]
                        loc = data['CoP coordinates'][stomp_start:stand_end]
                    else:
                        loc = data['CoP coordinates'][stomp_start:stomp_end]

                    remove = []
                    for j in range(loc.shape[0]):
                        if np.isnan(loc[j]).any():
                            remove.append(j)

                    loc = np.delete(loc, remove, axis=0)

                    jumping_CoPs[trial][foot][participant] = loc

    return jumping_CoPs

def walking_CoP_variation_plots_on_foot(step_CoPs):
    """ Plot mean and SD of CoP locations across all participants on the standardized foot shape for walking trials

    Args:
        step_CoPs: Dataframe containign CoP coordinates

    Returns:

    """
    trials = ['trial07', 'trial08', 'trial09', 'trial13', 'trial14', 'trial15', 'trial19']

    participant_colors = {'PPT_001': 'blue',
              'PPT_002': None,
              'PPT_003': None,
              'PPT_004': 'orange',
              'PPT_005': 'green',
              'PPT_006': None,
              'PPT_007': 'red',
              'PPT_008': 'black',
              'PPT_009': 'grey',
              'PPT_010': 'purple',
              'PPT_011': 'yellow',
              'PPT_012': None,
              'PPT_013': 'pink',
              'PPT_014': 'crimson',
              'PPT_015': 'mediumseagreen',
              'PPT_016': 'cyan',
              'PPT_017': 'magenta',
              'PPT_018': 'gold',
              'PPT_019': 'skyblue',
              'PPT_020': 'pink'}

    i = 0
    for trial in step_CoPs:
        i += 1

        j = 0
        for participant in step_CoPs[trial]['left']:

            if calibration_type[participant] == 'point':
                continue
            else:

                loc_y = 0
                loc_x = 0
                for foot in feet:
                    loc_y += np.nanmean(step_CoPs[trial][foot][participant][:, 1])
                    loc_x += np.nanmean(step_CoPs[trial][foot][participant][:, 0])

                loc_y = loc_y / 2
                loc_x = loc_x / 2

                plt.subplot(121)
                plt.scatter(i, loc_y, c=participant_colors[participant], label=participant)
                plt.xticks(ticks=list(range(1, len(trials) + 1)), labels=trials, rotation=80)

                plt.subplot(122)
                plt.scatter(loc_x, i, c=participant_colors[participant], label=participant)
                plt.xticks(ticks=list(range(1, len(trials) + 1)), labels=trials, rotation=80)

            j += 1

    sns.despine()
    plt.savefig(
        '../individual_figures/CoP_location_ppt.png',
        dpi=100)

    plt.figure(figsize=(15, 4))
    x_outline, y_outline = get_foot_outline()
    i=0
    for trial in step_CoPs:
        i += 1

        for participant in step_CoPs[trial]['left']:

            if calibration_type[participant] == 'point':
                continue
            else:

                loc_y = 0
                loc_x = 0
                for foot in feet:
                    loc_y += np.nanmean(step_CoPs[trial][foot][participant][:, 1])
                    loc_x += np.nanmean(step_CoPs[trial][foot][participant][:, 0])

                loc_y = loc_y / 2
                loc_x = loc_x / 2

                plt.subplot(1, len(list(step_CoPs.keys())), i)
                plt.title(trial)
                plt.scatter(x_outline, y_outline, s=2., c='black')
                plt.scatter(loc_x, loc_y, c=participant_colors[participant], label=participant, s=8)
                plt.axis('off')
                if i == 1:
                    plt.legend(bbox_to_anchor=(-.05, 1))
    plt.savefig('../individual_figures/walking_participant_average_CoP_on_foot.png')

    trial_average_x = np.zeros(len(step_CoPs.keys()))
    trial_average_y = np.zeros(len(step_CoPs.keys()))
    trial_std_x = np.zeros(len(step_CoPs.keys()))
    trial_std_y = np.zeros(len(step_CoPs.keys()))

    trials = ['trial09', 'trial07', 'trial08', 'trial15', 'trial13', 'trial14', 'trial19']
    tasks = ['Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking', 'Stairs', 'Gravel walking']
    selected_trial_colors = ['mediumaquamarine', 'limegreen', 'lime', 'forestgreen', 'mediumseagreen', 'chartreuse',
                    'greenyellow']

    i = 0
    for trial in trials:

        locs = np.zeros((0, 2))

        for foot in step_CoPs[trial]:

            for participant in step_CoPs[trial][foot]:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    locs = np.vstack((locs, step_CoPs[trial][foot][participant]))

        trial_average_x[i] = np.nanmean(locs[:, 0])
        trial_average_y[i] = np.nanmean(locs[:, 1])
        trial_std_x[i] = np.nanstd(locs[:, 0])
        trial_std_y[i] = np.nanstd(locs[:, 1])

        i += 1

    x_outline, y_outline = get_foot_outline()

    plt.figure(figsize=(18, 6))
    for i in range(len(trials)):
        plt.subplot(1, len(trials), i + 1)
        plt.scatter(x_outline, y_outline, c='black', s=2)
        plt.errorbar(x=trial_average_x[i], y=trial_average_y[i], xerr=trial_std_x[i], \
                     yerr=trial_std_y[i],
                     ecolor=selected_trial_colors[i], linestyle='')
        plt.scatter(x=trial_average_x[i], y=trial_average_y[i], c=selected_trial_colors[i])
        plt.axis('off')

    plt.savefig(
        '../individual_figures/walking_CoP_variation_foot.png',
        dpi=100)

def standing_trial_CoP_variation_plots(standing_CoPs):
    """ Plot mean and SD of CoP locations across all participants on the standardized foot shape for standing trials

    Args:
        standing_CoPs: Dataframe containign CoP coordinates

    Returns:

    """
    participant_colors = {'PPT_001': 'blue',
              'PPT_002': None,
              'PPT_003': None,
              'PPT_004': 'orange',
              'PPT_005': 'green',
              'PPT_006': None,
              'PPT_007': 'red',
              'PPT_008': 'black',
              'PPT_009': 'grey',
              'PPT_010': 'purple',
              'PPT_011': 'yellow',
              'PPT_012': None,
              'PPT_013': 'pink',
              'PPT_014': 'crimson',
              'PPT_015': 'mediumseagreen',
              'PPT_016': 'cyan',
              'PPT_017': 'magenta',
              'PPT_018': 'gold',
              'PPT_019': 'skyblue',
              'PPT_020': 'pink'}

    trial_ids = ['trial05', 'trial06', 'trial12', 'trial10', 'trial11']
    tasks = ['Quiet standing','Wobble-board','Twisting','Sit-to-stand','Sit-to-walk']

    x_outline, y_outline = get_foot_outline()

    trial_average_x = np.zeros(len(standing_CoPs.keys()))
    trial_average_y = np.zeros(len(standing_CoPs.keys()))
    trial_std_x = np.zeros(len(standing_CoPs.keys()))
    trial_std_y = np.zeros(len(standing_CoPs.keys()))

    i = 0
    for trial in standing_CoPs:

        locs = np.zeros((0, 2))

        for foot in standing_CoPs[trial]:

            for participant in standing_CoPs[trial][foot]:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    locs = np.vstack((locs, standing_CoPs[trial][foot][participant]))

        trial_average_x[i] = np.nanmean(locs[:, 0])
        trial_average_y[i] = np.nanmean(locs[:, 1])
        trial_std_x[i] = np.nanstd(locs[:, 0])
        trial_std_y[i] = np.nanstd(locs[:, 1])

        i += 1

    selected_trial_colors = ['indigo', 'purple', 'mediumorchid', 'rebeccapurple', 'blueviolet']

    plt.figure(figsize=(10,6))
    for i in range(len(selected_trial_colors)):
        plt.subplot(1, len(selected_trial_colors), i + 1)
        plt.scatter(x_outline, y_outline, c='black', s=2)
        plt.errorbar(x=trial_average_x[i], y=trial_average_y[i], xerr=trial_std_x[i], \
                     yerr=trial_std_y[i],
                     ecolor=selected_trial_colors[i], linestyle='')
        plt.scatter(x=trial_average_x[i], y=trial_average_y[i], c=selected_trial_colors[i])
        plt.axis('off')

    plt.savefig(
        '../individual_figures/standing_CoP_variation_foot.png',
        dpi=100)

    plt.figure()
    i = 0
    for trial in standing_CoPs:
        i += 1

        j = 0
        for participant in standing_CoPs[trial]['left']:

            if calibration_type[participant] == 'point':
                continue
            else:

                loc_y = 0
                loc_x = 0
                for foot in feet:
                    loc_y += np.nanmean(standing_CoPs[trial][foot][participant][:, 1])
                    loc_x += np.nanmean(standing_CoPs[trial][foot][participant][:, 0])

                loc_y = loc_y / 2
                loc_x = loc_x / 2

                plt.subplot(121)
                plt.scatter(i, loc_y, c=participant_colors[participant], label=participant)
                plt.xticks(ticks=list(range(1, len(trial_ids) + 1)), labels=tasks, rotation=80)

                plt.subplot(122)
                plt.scatter(loc_x, i, c=participant_colors[participant], label=participant)
                plt.yticks(ticks=list(range(1, len(trial_ids) + 1)), labels=tasks, rotation=80)

            j += 1

    sns.despine()
    plt.savefig(
        '../individual_figures/CoP_location_ppt_standing.png',
        dpi=100)

    plt.figure(figsize=(15, 4))
    x_outline, y_outline = get_foot_outline()
    i=0
    for trial in standing_CoPs:
        i += 1

        for participant in standing_CoPs[trial]['left']:

            if calibration_type[participant] == 'point':
                continue
            else:

                loc_y = 0
                loc_x = 0
                for foot in feet:
                    loc_y += np.nanmean(standing_CoPs[trial][foot][participant][:, 1])
                    loc_x += np.nanmean(standing_CoPs[trial][foot][participant][:, 0])

                loc_y = loc_y / 2
                loc_x = loc_x / 2

                plt.subplot(1, len(list(standing_CoPs.keys())), i)
                plt.title(trial)
                plt.scatter(x_outline, y_outline, s=2., c='black')
                plt.scatter(loc_x, loc_y, c=participant_colors[participant], label=participant, s=8.)
                plt.axis('off')
                if i == 1:
                    plt.legend(bbox_to_anchor=(-.05, 1))
    plt.savefig('../individual_figures/standing_participant_average_CoP_on_foot.png')

def jumping_trial_CoP_variation_plots(jumping_CoPs):
    """ Plot mean and SD of CoP locations across all participants on the standardized foot shape for jumping trials

    Args:
        jumping_CoPs: Dataframe containign CoP coordinates

    Returns:

    """

    participant_colors = {'PPT_001': 'blue',
              'PPT_002': None,
              'PPT_003': None,
              'PPT_004': 'orange',
              'PPT_005': 'green',
              'PPT_006': None,
              'PPT_007': 'red',
              'PPT_008': 'black',
              'PPT_009': 'grey',
              'PPT_010': 'purple',
              'PPT_011': 'yellow',
              'PPT_012': None,
              'PPT_013': 'pink',
              'PPT_014': 'crimson',
              'PPT_015': 'mediumseagreen',
              'PPT_016': 'cyan',
              'PPT_017': 'magenta',
              'PPT_018': 'gold',
              'PPT_019': 'skyblue',
              'PPT_020': 'pink'}

    trial_ids = ['trial16','trial17','trial18']
    tasks = ['Both feet jump','Left foot jump','Right foot jump']

    x_outline, y_outline = get_foot_outline()

    trial_average_x = np.zeros(len(jumping_CoPs.keys()))
    trial_average_y = np.zeros(len(jumping_CoPs.keys()))
    trial_std_x = np.zeros(len(jumping_CoPs.keys()))
    trial_std_y = np.zeros(len(jumping_CoPs.keys()))

    i = 0
    for trial in jumping_CoPs:

        locs = np.zeros((0, 2))

        for foot in jumping_CoPs[trial]:

            for participant in jumping_CoPs[trial][foot]:

                if calibration_type[participant] == 'point':
                    continue
                else:

                    locs = np.vstack((locs, jumping_CoPs[trial][foot][participant]))

        trial_average_x[i] = np.nanmean(locs[:, 0])
        trial_average_y[i] = np.nanmean(locs[:, 1])
        trial_std_x[i] = np.nanstd(locs[:, 0])
        trial_std_y[i] = np.nanstd(locs[:, 1])

        i += 1

    selected_trial_colors = ['crimson', 'indianred', 'red']

    plt.figure(figsize=(10,6))
    for i in range(len(selected_trial_colors)):
        plt.subplot(1, len(selected_trial_colors), i + 1)
        plt.scatter(x_outline, y_outline, c='black', s=2)
        plt.errorbar(x=trial_average_x[i], y=trial_average_y[i], xerr=trial_std_x[i], \
                     yerr=trial_std_y[i],
                     ecolor=selected_trial_colors[i], linestyle='')
        plt.scatter(x=trial_average_x[i], y=trial_average_y[i], c=selected_trial_colors[i])
        plt.axis('off')

    plt.savefig(
        '../individual_figures/jumping_CoP_variation_foot.png',
        dpi=100)

    plt.figure()
    i = 0
    for trial in jumping_CoPs:
        i += 1

        j = 0
        for participant in jumping_CoPs[trial]['left']:

            if calibration_type[participant] == 'point':
                continue
            else:

                loc_y = 0
                loc_x = 0
                for foot in feet:
                    loc_y += np.nanmean(jumping_CoPs[trial][foot][participant][:, 1])
                    loc_x += np.nanmean(jumping_CoPs[trial][foot][participant][:, 0])

                loc_y = loc_y / 2
                loc_x = loc_x / 2

                plt.subplot(121)
                plt.scatter(i, loc_y, c=participant_colors[participant], label=participant)
                plt.xticks(ticks=list(range(1, len(trial_ids) + 1)), labels=tasks, rotation=80)

                plt.subplot(122)
                plt.scatter(loc_x, i, c=participant_colors[participant], label=participant)
                plt.yticks(ticks=list(range(1, len(trial_ids) + 1)), labels=tasks, rotation=80)

            j += 1

    sns.despine()
    plt.savefig(
        '../individual_figures/CoP_location_ppt_jumping.png',
        dpi=100)

    plt.figure(figsize=(15, 4))
    x_outline, y_outline = get_foot_outline()
    i=0
    for trial in jumping_CoPs:
        i += 1

        for participant in jumping_CoPs[trial]['left']:

            if calibration_type[participant] == 'point':
                continue
            else:

                loc_y = 0
                loc_x = 0
                for foot in feet:
                    loc_y += np.nanmean(jumping_CoPs[trial][foot][participant][:, 1])
                    loc_x += np.nanmean(jumping_CoPs[trial][foot][participant][:, 0])

                loc_y = loc_y / 2
                loc_x = loc_x / 2

                plt.subplot(1, len(list(jumping_CoPs.keys())), i)
                plt.title(trial)
                plt.scatter(x_outline, y_outline, s=2., c='black')
                plt.scatter(loc_x, loc_y, c=participant_colors[participant], label=participant, s=8.)
                plt.axis('off')
                if i == 1:
                    plt.legend(bbox_to_anchor=(-.05, 1))
    plt.savefig('../individual_figures/jumping_participant_average_CoP_on_foot.png')


def CoP_analysis_during_steps_PPT_level(normalized_steps):
    """ Calculate the average centre of pressure location, and deviations from this
        for each of the walking trials

        :param filepath_prefix:
        :param output_path_prefix:
        :return:
        """

    trials = ['trial07', 'trial08', 'trial09', 'trial13', 'trial14', 'trial15', 'trial19']

    all_CoPs = {}


    for trial in trials:
        all_CoPs[trial] = {}

        for foot in feet:
            all_CoPs[trial][foot] = {}

            for participant in details[trial]:

                if calibration_type[participant] == 'point':
                   continue
                else:

                    all_CoPs[trial][foot][participant] = np.zeros((0, 2))

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

                        loc, x, y = centre_of_pressure(D, threshold=100)

                        remove = []
                        for j in range(loc.shape[0]):
                            if np.isnan(loc[j]).any():
                                remove.append(j)

                        loc = np.delete(loc, remove, axis=0)

                    all_CoPs[trial][foot][participant] = np.vstack((all_CoPs[trial][foot][participant], loc))

    return all_CoPs


def NMF_component_clusters(dfs, models, scaled):
    """ Use K-means clustering on the first 5 components in each task.

    Args:
        dfs (dict): dictionary containing sorted dataframes
        models (dict): dictionary containing NMF component and weight matrices
        scaled (dict): dictionary containing all participant data scaled to a uniform size per task

    Returns:

    """

    new_H = {}
    coords = np.zeros((0, 2))

    plt.figure(figsize=(30, 60))
    for trial in models:
        H = models[trial]['H']
        s = scaled[trial]['Stimulus']

        li = np.asarray(dfs[trial]['Component'])

        H_new = np.zeros((0, H.shape[1]))

        for k in range(5):
            H_new = np.vstack((H_new, H[li[k], :]))

        new_H[trial] = H_new

        for k in range(5):
            ind = np.argpartition(H_new[k], - int(len(H_new[k]) * .025))[-int(len(H_new[k]) * .025):]

            x_ = s.location[ind, 0]
            y_ = s.location[ind, 1]

            points = np.vstack((x_, y_)).T

            coords = np.vstack((coords, points))

        if trial == 'trial05':
            df = pd.DataFrame(new_H[trial])
            trial_id = trial[-2:]
            df.index = [str(trial_id) + '.1', str(trial_id) + '.2', str(trial_id) + '.3', \
                        str(trial_id) + '.4', str(trial_id) + '.5']
            df_ = df
        else:

            df = pd.DataFrame(new_H[trial])
            trial_id = trial[-2:]
            df.index = [str(trial_id) + '.1', str(trial_id) + '.2', str(trial_id) + '.3', \
                        str(trial_id) + '.4', str(trial_id) + '.5']
            df_ = df_.append(df)

    plt.figure(figsize=(15, 15))
    sns.heatmap(df_.T.corr(), square=True)
    plt.savefig(
        '../individual_figures/NMF_first_5_correlations.png',
        dpi=100)

    # Import ElbowVisualizer
    plt.close()
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2, 30), timings=True)
    visualizer.fit(coords)  # Fit data to visualizer
    n_clusters = visualizer.elbow_value_  # Finalize and render figure


    x_outline, y_outline = get_foot_outline()

    model = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(coords)

    cluster_colors = ['green', 'blue', 'black', 'red', 'pink', 'orange', 'cyan']

    plt.figure(figsize=(3, 7))
    for i in range(n_clusters):
        x__ = coords[y_clusters == i, 0]
        y__ = coords[y_clusters == i, 1]

        points = np.vstack((x__, y__)).T

        hull = ConvexHull(points)

        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'k', alpha=0.3, c=cluster_colors[i],
                 label='Cluster ' + str(i + 1))
    plt.scatter(x_outline, y_outline, s=3, color='black')
    plt.legend(bbox_to_anchor=(-.05, 1))
    plt.axis('off')
    plt.savefig(
        '../individual_figures/NMF_clusters.png',
        dpi=100, bbox_inches='tight')


def force_over_time(step_all_metrics):
    """ Plot lineplot of force experienced over time in all walking tasks

    Args:
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:

    """
    plt.rcParams.update({'font.size': 50})
    plt.figure(figsize=(40, 20), dpi=100)
    sns.relplot(data=step_all_metrics, x='Timepoints', y='Force', col='Trial title', \
                kind='line', ci=95, hue='Participant', alpha=.5, \
                col_order=['Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking', 'Stairs', 'Gravel walking'])
    plt.xlim(0,100)
    plt.ylim(0,400)
    plt.savefig('../individual_figures/force_over_time.png', dpi=100)


def area_over_time(step_all_metrics):
    """ Plot lineplot of contact area experienced over time in all walking tasks

    Args:
        step_all_metrics: dataframe containing analysed data for walking tasks calculated on normalized step lengths

    Returns:

    """
    plt.rcParams.update({'font.size': 50})
    plt.figure(figsize=(40, 20), dpi=100)
    sns.relplot(data=step_all_metrics, x='Timepoints', y='Contact area percent', col='Trial title', \
                kind='line', ci=95, hue='Participant', alpha=.5, \
                col_order=['Slow walking', 'Normal walking', 'Fast walking', 'Jogging', 'Slope walking', 'Stairs', 'Gravel walking'])
    plt.xlim(0, 100)
    plt.ylim(0,100)
    plt.savefig('../individual_figures/area_over_time.png', dpi=100)


def trial_type_component_clusters(trial_type_dfs, trial_type_models, trial_type_scaled):
    """ Use K-means clustering on the first 10 components in each task.

    Args:
        trial_type_dfs (dict): dictionary containing sorted dataframes for task type
        trial_type_models (dict): dictionary containing NMF component and weight matrices for task types
        trial_type_scaled (dict): dictionary containing all participant data scaled to a uniform size per task type

    Returns:

    """

    x_outline, y_outline = get_foot_outline()

    trial_type_dfs = sort_nmf_df(trial_type_dfs)

    colors = ['green', 'blue', 'black', 'red', 'pink', 'orange', 'cyan']

    for trial_type in trial_type_dfs:
        coords = np.zeros((0, 2))

        H = trial_type_models[trial_type]['H']
        s = trial_type_scaled[trial_type]['Stimulus']

        li = np.asarray(trial_type_dfs[trial_type]['Component'])

        H_new = np.zeros((0, H.shape[1]))

        for k in range(10):
            H_new = np.vstack((H_new, H[li[k], :]))

        for k in range(10):
            ind = np.argpartition(H_new[k], - int(len(H_new[k]) * .025))[-int(len(H_new[k]) * .025):]

            x_ = s.location[ind, 0]
            y_ = s.location[ind, 1]

            points = np.vstack((x_, y_)).T

            coords = np.vstack((coords, points))

        model = KMeans()
        # k is range of number of clusters.
        visualizer = KElbowVisualizer(model, k=(2, 30), timings=True)
        visualizer.fit(coords)  # Fit data to visualizer
        n_clusters = visualizer.elbow_value_  # Finalize and render figure

        model = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
        y_clusters = model.fit_predict(coords)

        plt.figure(figsize=(4, 7))
        plt.title(trial_type, fontsize=30)
        plt.scatter(x_outline, y_outline, s=3, color='black')
        for i in range(n_clusters):
            x__ = coords[y_clusters == i, 0]
            y__ = coords[y_clusters == i, 1]

            points = np.vstack((x__, y__)).T

            hull = ConvexHull(points)

            plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'k', alpha=0.3, c=colors[i],
                     label='Component ' + str(i + 1))

        plt.legend(bbox_to_anchor=(-.05, 1))
        plt.axis('off')
        plt.savefig('../individual_figures/' + trial_type + '_clusters.png', dpi=600, bbox_inches='tight')



def trial_type_nmf_components_required_on_foot(trial_type_dfs, trial_type_models, trial_type_scaled):
    """ Plot first 15 components for each task tupe

    Args:
        dfs (dict): dictionary containing sorted dataframes
        models (dict): dictionary containing NMF component and weight matrices
        scaled (dict): dictionary containing all participant data scaled to a uniform size per task type

    Returns:

    """

    x_outline, y_outline = get_foot_outline()

    components_required = {}

    for i in range(len(list(trial_type_dfs.keys()))):
        W = trial_type_models[list(trial_type_dfs.keys())[i]]['W']
        H = trial_type_models[list(trial_type_dfs.keys())[i]]['H']

        li = np.asarray(trial_type_dfs[list(trial_type_dfs.keys())[i]]['Component'])

        array = np.asarray(trial_type_dfs[list(trial_type_dfs.keys())[i]]['Cumulative variance explained'])
        for j in range(len(array)):
            if array[j] > .8:
                num = j
                break

        components_required[list(trial_type_dfs.keys())[i]] = num

        W_new = np.zeros((0, W.shape[0]))
        H_new = np.zeros((0, H.shape[1]))

        for k in range(15):
            W_new = np.vstack((W_new, W[:, li[k]]))
            H_new = np.vstack((H_new, H[li[k], :]))

        s = trial_type_scaled[list(trial_type_dfs.keys())[i]]['Stimulus']

        plt.figure(figsize=(3 * 15, 6))
        for l in range(H_new[:15].shape[0]):
            plt.subplot(1, 15, l + 1)
            plt.title(trial_types[i] + ' \n Component ' + str(li[l]))
            plt.scatter(s.location[:, 0], s.location[:, 1],
                        c=H_new[l], cmap='RdPu', marker=",", vmin=0)
            plt.scatter(x_outline, y_outline, s=3., c='black')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(
            '../individual_figures/nmf_components_' +
            trial_types[i] + '.png',
            dpi=600)

        plt.figure(figsize=(3 * 5, 6))
        for l in range(H_new[:5].shape[0]):
            plt.subplot(1, 5, l + 1)
            plt.title(trial_types[i] + ' \n Component ' + str(li[l]))
            plt.scatter(s.location[:, 0], s.location[:, 1],
                        c=H_new[l], cmap='RdPu', marker=",", vmin=0)
            plt.scatter(x_outline, y_outline, s=3., c='black')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(
            '../individual_figures/5_nmf_components_' +
            trial_types[i] + '.png',
            dpi=600)



def contact_area_probability(scaled_by_trial):
    """ Plot probability that each sensor will be active across all tasks

    Args:
        scaled_by_trial (dict): dictionary containing all participant data scaled to a uniform size per task

    Returns:

    """
    sorted_trial_ids = ['trial05','trial06','trial12','trial10','trial11',\
                        'trial09','trial07','trial08','trial15','trial13','trial14','trial19',\
                        'trial16','trial17','trial18']

    x_outline, y_outline = get_foot_outline()

    k=0
    plt.figure(figsize=(45, 6))
    for trial in sorted_trial_ids:
        k += 1

        data = scaled_by_trial[trial]['Raw data']
        s = scaled_by_trial[trial]['Stimulus']

        data[data < 1.] = 0
        data[data > 1] = 1
        data = np.mean(data, axis=2)
        data = np.dstack((data, data))
        data = reshape_data(data, scaled_by_trial[trial]['idxs'])

        plt.subplot(1, len(sorted_trial_ids), k)
        plt.title(trial_titles[trial])
        plt.scatter(s.location[:, 0], s.location[:, 1], c=data[:, 0], marker=",", cmap='turbo',
                    vmin=0, vmax=1)
        plt.scatter(x_outline, y_outline, c='black', s=5.)
        plt.axis('off')
    plt.savefig(
        '../individual_figures/contact_binary.png',
        dpi=100)


def contact_area_probability_all_data(scaled):
    """ Plot probability that each sensor will be active across all tasks

    Args:
        scaled (dict): dictionary containing all participant data scaled to a uniform size across all tasks

    Returns:

    """
    x_outline, y_outline = get_foot_outline()


    data = scaled['Raw data']
    s = scaled['Stimulus']

    data[data < 1.] = 0
    data[data > 1] = 1
    data = np.mean(data, axis=2)
    data = np.dstack((data, data))
    data = reshape_data(data, scaled['idxs'])

    plt.figure(figsize=(2.5, 6), dpi=100)
    plt.title('Entire dataset')
    # plt.imshow(np.mean(data, axis=2), vmin=0,\
    #           extent=[x_outline.min(), x_outline.max(), \
    #                   y_outline.min(), y_outline.max()], cmap='inferno')
    # plt.scatter(x_outline, y_outline, c='white', s=1.)
    plt.scatter(s.location[:, 0], s.location[:, 1], c=data[:, 0], marker=",", cmap='turbo',
                vmin=0, vmax=1)
    plt.scatter(x_outline, y_outline, c='black', s=3.)
    plt.axis('off')
    plt.savefig(
        '../individual_figures/contact_binary_all_data.png',
        dpi=100, bbox_inches='tight')


def individual_frame_contact(scaled_by_trial):
    """ Randomly pick 10 individual frames of data and plot contact probability (0 and 1)

    Args:
        scaled_by_trial (dict): dictionary containing all participant data scaled to a uniform size per task

    Returns:

    """
    x_outline, y_outline = get_foot_outline()

    sorted_trial_ids = ['trial05','trial06','trial12','trial10','trial11',\
                        'trial09','trial07','trial08','trial15','trial13','trial14','trial19',\
                        'trial16','trial17','trial18']

    k = 0
    plt.figure(figsize=(13, 40))
    for trial in sorted_trial_ids[5:-1]:
        data = scaled_by_trial[trial]['Raw data']
        s = scaled_by_trial[trial]['Stimulus']
        idxs = scaled_by_trial[trial]['idxs']

        for i in range(10):
            plt.subplot(len(sorted_trial_ids), 10, (k * 10) + (i + 1))
            frame_to_show = random.randint(0, data.shape[2])

            d = data[:, :, frame_to_show]
            d[d < 1.] = 0
            d[d > 1] = 1

            d = np.dstack((d, d))
            d = reshape_data(d, idxs)

            plt.scatter(s.location[:, 0], s.location[:, 1], c=d[:, 0], marker=",", cmap='bwr',
                        vmin=0, vmax=1)

            if i == 9:
                plt.colorbar(ticks=[0, 1])
            plt.scatter(x_outline, y_outline, c='black', s=5.)
            plt.axis('off')
            if i == 0:
                plt.ylabel(trial_titles[trial], fontsize=10)

        k += 1
    plt.savefig(
        '../individual_figures/contact_binary_single_frames.png',
        dpi=100, bbox_inches='tight')




def CoP_CoC_distance_bars(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates):
    """ Calculate the distance between the CoP and CoC coordinates

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

    all_metrics = all_metrics[all_metrics['Force'] > 5.]
    all_CoP_coordinates = all_CoP_coordinates[all_CoP_coordinates['Force'] > 5.]


    ### locomotion trials

    step_CoP_coordinates['X_distance'] = np.abs((step_CoP_coordinates['X_contact'] - step_CoP_coordinates['X_pressure'])) / (x_outline.max() - x_outline.min()) * 100
    step_CoP_coordinates['Y_distance'] = np.abs((step_CoP_coordinates['Y_contact'] - step_CoP_coordinates['Y_pressure'])) / (y_outline.max() - y_outline.min()) * 100

    step_CoP_coordinates = step_CoP_coordinates.reset_index()
    step_all_metrics = step_all_metrics.reset_index()

    forces = step_all_metrics['Force'].copy()
    step_CoP_coordinates['Force'] = forces.values

    step_all_metrics = step_all_metrics[step_all_metrics['Force'] > 5.]
    step_CoP_coordinates = step_CoP_coordinates[step_CoP_coordinates['Force'] > 5.]


    distance_df = pd.concat([step_CoP_coordinates, all_CoP_coordinates])
    distance_df.to_csv('/Users/lukecleland/Downloads/csv.csv')

    # bins of 2.5%
    bins = np.arange(0, 50, 2.5)

    distance_x = distance_df.copy()
    distance_x['binned'] = np.searchsorted(bins, distance_x['X_distance'].values)
    binned_x = distance_x.groupby(['Trial', pd.cut(distance_x['X_distance'], bins=bins)]).size()
    #binned_x.to_csv('/Users/lukecleland/Downloads/x_binned.csv')
    binned_x = pd.DataFrame(binned_x).reset_index().rename(columns={0: 'Count'})  # rename column

    distance_y = distance_df.copy()
    distance_y['binned'] = np.searchsorted(bins, distance_y['Y_distance'].values)
    binned_y = distance_y.groupby(['Trial', pd.cut(distance_y['Y_distance'], bins=bins)]).size()
    #binned_y.to_csv('/Users/lukecleland/Downloads/y_binned.csv')
    binned_y = pd.DataFrame(binned_y).reset_index().rename(columns={0: 'Count'}) # rename column

    plotting_df_x = pd.DataFrame({'Bins': [2.5]*(len(bins)-1)})
    plotting_df_y = pd.DataFrame({'Bins': [2.5] * (len(bins) - 1)})

    # x axis
    for trial in pd.unique(binned_x['Trial']):
        trial_df = binned_x[binned_x['Trial'] == trial]

        total = trial_df['Count'].sum()
        trial_df['Percentage'] = trial_df['Count'] / total

        percent = trial_df['Percentage'].values

        plotting_df_x[trial] = percent

    normalized_x = plotting_df_x.copy()
    normalized_x = normalized_x.drop(columns='Bins')
    normalized_x[(normalized_x > 0.) & (normalized_x < .01)] = .01
    normalized_x = (normalized_x - 0) / (normalized_x.max().max() - 0) # normalize between 0 and max in df

    # y axis
    for trial in pd.unique(binned_y['Trial']):
        trial_df = binned_y[binned_y['Trial'] == trial]

        total = trial_df['Count'].sum()
        trial_df['Percentage'] = trial_df['Count'] / total

        percent = trial_df['Percentage'].values

        plotting_df_y[trial] = percent

    normalized_y = plotting_df_y.copy()
    normalized_y = normalized_y.drop(columns='Bins')
    normalized_y[(normalized_y > 0.) & (normalized_y < .01)] = .01
    normalized_y = (normalized_y - 0) / (normalized_y.max().max() - 0)  # normalize between 0 and max in df



    bin_limits = trial_df['Y_distance'].values

    fig = plt.figure(constrained_layout=True, figsize=(30, 40), dpi=100)
    plt.rcParams.update({'font.size': 40})
    gs = GridSpec(2, 3, figure=fig, height_ratios=[.85, .85])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])

    sorted_trials = ['trial05', 'trial06', 'trial12', 'trial10', 'trial11', \
                     'trial09', 'trial07', 'trial08', 'trial15', 'trial13', 'trial14', 'trial19', \
                     'trial16', 'trial17', 'trial18']
    width = 0.75  # the width of the bars: can also be len(x) sequence
    bottom = np.arange(0, 47.5, 2.5)
    for j in range(len(sorted_trials)):
        for i in range(len(bottom)):

            ax1.bar(trial_titles[sorted_trials[j]], 2.5, width, label=bin_limits[i], bottom=bottom[i],
                   color=trial_colors[sorted_trials[j]],
                   alpha=normalized_y[sorted_trials[j]][i])
            if normalized_y[sorted_trials[j]][i] > 0.:
                ax1.bar(trial_titles[sorted_trials[j]], 2.5, width, label=bin_limits[i], bottom=bottom[i], color='none',
                       edgecolor='black')

    ax1.set_ylim(0, 50)
    ax1.set_title('Distance between CoP and centre of contact \n along anterior-posterior axis', fontsize=50)
    ax1.set_ylabel('Distance (% foot length)', fontsize=40)
    ax1.tick_params(axis='y', labelsize=35)
    ax1.set_xlabel('Task', fontsize=40)
    ax1.tick_params(axis='x', labelsize=35, labelrotation=80)
    sns.despine(ax=ax1)


    for j in range(len(sorted_trials)):
        for i in range(len(bottom)):

            ax2.bar(trial_titles[sorted_trials[j]], 2.5, width, label=bin_limits[i], bottom=bottom[i],
                   color=trial_colors[sorted_trials[j]],
                   alpha=normalized_x[sorted_trials[j]][i])
            if normalized_x[sorted_trials[j]][i] > 0.:
                ax2.bar(trial_titles[sorted_trials[j]], 2.5, width, label=bin_limits[i], bottom=bottom[i], color='none',
                       edgecolor='black')

    ax2.set_ylim(0, 50)
    ax2.set_title('Distance between CoP and centre of contact \n along medial-lateral axis', fontsize=50)
    ax2.set_ylabel('Distance (% foot width)', fontsize=40)
    ax2.tick_params(axis='y', labelsize=35)
    ax2.set_xlabel('Task', fontsize=40)
    ax2.tick_params(axis='x', labelsize=35, labelrotation=80)
    sns.despine(ax=ax2)

    plt.savefig('../individual_figures/CoP_CoC_distance.png', dpi=100)


def correlation_between_CoP_CoC(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates):
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
    print('X R2: ', stats.spearmanr(distance_df['X_pressure'], distance_df['X_contact'])[0]**2)
    print('Y R2: ', stats.spearmanr(distance_df['Y_pressure'], distance_df['Y_contact'])[0]**2)

    x_corr = stats.spearmanr(distance_df['X_pressure'], distance_df['X_contact'])[0]
    y_corr = stats.spearmanr(distance_df['Y_pressure'], distance_df['Y_contact'])[0]

    plt.figure(figsize=(10,10), dpi=100)

    plt.scatter(x_corr**2, y_corr**2, color='black', s=80, marker=',', label='All data')

    for trial_type in pd.unique(distance_df['Trial type']):
        trial_df = distance_df[distance_df['Trial type'] == trial_type]

        print(trial_type)
        print('X R2: ', stats.spearmanr(trial_df['X_pressure'], trial_df['X_contact'])[0]**2)
        print('Y R2: ', stats.spearmanr(trial_df['Y_pressure'], trial_df['Y_contact'])[0]**2)

        x_corr = stats.spearmanr(trial_df['X_pressure'], trial_df['X_contact'])[0]
        y_corr = stats.spearmanr(trial_df['Y_pressure'], trial_df['Y_contact'])[0]

        plt.scatter(x_corr**2, y_corr**2, color=trial_type_colors[trial_type], s=90, marker='*', label=trial_type)

    for trial in pd.unique(distance_df['Trial']):
        trial_df = distance_df[distance_df['Trial'] == trial]

        print(trial)
        print('X R2: ', stats.spearmanr(trial_df['X_pressure'], trial_df['X_contact'])[0]**2)
        print('Y R2: ', stats.spearmanr(trial_df['Y_pressure'], trial_df['Y_contact'])[0]**2)

        x_corr = stats.spearmanr(trial_df['X_pressure'], trial_df['X_contact'])[0]
        y_corr = stats.spearmanr(trial_df['Y_pressure'], trial_df['Y_contact'])[0]

        plt.scatter(x_corr**2, y_corr**2, color=trial_colors[trial], s=80, marker='o', label=trial_titles[trial])

    line_coords = [0.65, .98]
    plt.plot(line_coords, line_coords, color='lightgrey')

    plt.xlabel('Medial-lateral axis', fontsize=40)
    plt.ylabel('Anterior-posterior axis', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(fontsize=35)
    plt.ylim(.65, .98)
    plt.xlim(.65, .98)
    plt.title('Variance explained between CoP and CoC', fontsize=50)
    sns.despine()
    plt.savefig('../individual_figures/CoP_CoC_correlation.png', dpi=100)


def CoP_CoC_scatterplot(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates):
    """ Plot two scatterplots - showing CoP and CoC coordinates along the two axis of the foot. Each axis plotted
        separately

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

    xs = [0, 25, 50, 75, 100, 125, 150]
    colors = ['red', 'darkorange', 'goldenrod', 'lime', 'cyan', 'dodgerblue', 'magenta']
    plt.figure(figsize=(20, 20), dpi=600)
    plt.subplot(221)
    plt.scatter(distance_df['X_pressure'], distance_df['X_contact'], c=distance_df['X_distance'], s=1.5,
                cmap='cubehelix', vmin=0, vmax=36)
    plt.xlim(x_outline.min(), x_outline.max())
    plt.ylim(x_outline.min(), x_outline.max())
    plt.scatter(xs, xs, c=colors, marker='x', s=100, lw=5)
    plt.xlabel('CoP along ML axis', fontsize=40)
    plt.ylabel('CoC along ML axis', fontsize=40)
    locs, labels = plt.xticks()
    plt.xticks([x_outline.min(), x_outline.max()], ['Lateral', 'Medial'])
    locs, labels = plt.yticks()
    plt.yticks([x_outline.min(), x_outline.max()], ['Lateral', 'Medial'])
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('Relationship between CoP and CoC estimates \n along the medial-lateral axis', fontsize=50)
    sns.despine()

    plt.subplot(222)
    plt.scatter(x_outline, y_outline, c='black', s=3.)
    plt.axis('off')
    # plt.ylim(y_outline.min()-10, y_outline.max()+10)
    plt.xlim(x_outline.min(), x_outline.max())
    for i in range(len(xs)):
        plt.axvline(xs[i], c=colors[i], lw=4)


    # AP axis
    xs = [-150, -50, 50, 150, 250, 350, 450]
    colors = ['red', 'darkorange', 'goldenrod', 'lime', 'cyan', 'dodgerblue', 'magenta']

    plt.subplot(223)
    plt.scatter(distance_df['Y_pressure'], distance_df['Y_contact'], c=distance_df['Y_distance'], s=1.5,
                cmap='cubehelix', vmin=0, vmax=60)
    plt.xlim(y_outline.min() - 10, y_outline.max() + 10)
    plt.ylim(y_outline.min() - 10, y_outline.max() + 10)
    plt.scatter(xs, xs, c=colors, marker='x', s=100, edgecolor='white', lw=5.)
    plt.xlabel('CoP along AP axis', fontsize=40)
    plt.ylabel('CoC along AP axis', fontsize=40)
    locs, labels = plt.xticks()
    plt.xticks([y_outline.min() - 10, y_outline.max() + 10], ['Posterior', 'Anterior'])
    locs, labels = plt.yticks()
    plt.yticks([y_outline.min() - 10, y_outline.max() + 10], ['Posterior', 'Anterior'])
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('Relationship between CoP and CoC estimates \n along the anterior-posterior axis', fontsize=50)
    sns.despine()

    plt.subplot(224)
    plt.scatter(x_outline, y_outline, c='black', s=3.)
    plt.axis('off')
    plt.ylim(y_outline.min() - 10, y_outline.max() + 10)
    for i in range(len(xs)):
        plt.axhline(xs[i], c=colors[i], lw=4)
    plt.savefig('../individual_figures/CoP_CoC_scatterplot.png', dpi=100)



def examples_of_individual_frames():
    """ Generates figures that show individual frames of pressure data

    Returns:

    """

    x_outline, y_outline = get_foot_outline()

    prefix = '../raw_data/'

    output_path_prefix = '../preprocessed_data/'

    prefix2 = '../preprocessed_data/'

    stomps = decompress_pickle('../processed_data/pressure_stomps.pbz2')
    idxs = decompress_pickle('../processed_data/participant_stimulus_indexes.pbz2')

    frames = {'trial05': {'PPT_010': 2504,
                          'PPT_009': 1706},
              'trial06': {'PPT_011': 3245,
                          'PPT_001': 3206,
                          'PPT_017': 2632,
                          'PPT_014': 1683},
              'trial07': {'PPT_018': 3231,
                          'PPT_018': 732},
              'trial15': {'PPT_009': 1009,
                          'PPT_005': 1712},
              'trial19': {'PPT_016': 3110,
                          # 'PPT_016': 1167,
                          # 'PPT_016': 1307
                          'PPT_004': 3695},
              'trial16': {'PPT_008': 211,
                          'PPT_010': 340}}

    frames = {'trial19': {  'PPT_016': 3110},
        # 'PPT_016': 1167,
    #    'PPT_016': 1307,
    #    'PPT_004': 3695},
    #    'trial07': {'PPT_018': 3251},
        'trial12': {'PPT_015': 760}}

    i = 0

    for trial in frames:

        for participant in frames[trial]:
            trial_number = trial[-2:]
            foot = 'left'
            suffix = 'L_M'

            filepath = prefix + participant + '/' + trial + '/' + dates[participant] + 'PPT' + str(
                trial_number) + suffix + '.csv'

            # read in raw data
            data = import_data(filepath=filepath, frames=frames_per_trial[participant][trial],
                               calibration_type=calibration_type[participant],
                               extended_calibration=extended_calibration[participant][trial])

            file = prefix2 + participant + '/' + trial + '/' + foot + '/' + foot + ' data.pbz2'
            data = decompress_pickle(file)
            processed = data['Raw data']
            reshaped = data['Reshaped data']
            s = data['Stimulus']

            stomp_start = stomps[trial][participant][0] + 5
            stomp_end = stomps[trial][participant][1] - 5

            frame = frames[trial][participant]

            i += 1
            plt.figure(figsize=(3, 5), dpi=600)
            # plt.imshow(processed[:, :, frame], extent=[x_outline.min(), x_outline.max(), y_outline.min(), y_outline.max()])
            plt.scatter(s.location[:, 0], s.location[:, 1], c=reshaped[:, frame], cmap='YlOrRd', marker=',')
            plt.scatter(x_outline, y_outline, c='black', s=2.)
            plt.axis('off')
            # plt.show()

            plt.savefig('../individual_figures/examples_of_raw_data/' + participant + '_' + trial + '_' + str(
                frames[trial][participant]) + '.png')





def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother  - extracted from https://wil.yegelwel.com/cluster-correlation-matrix/

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def component_correlation_matrix(dfs, models, scaled):
    """ Correlte the components required to explain 80% variance in each task with each other,
    and sorted to highlight clusters

    Args:
        dfs (dict): dictionary containing sorted dataframes
        models (dict): dictionary containing NMF component and weight matrices
        scaled (dict): dictionary containing all participant data scaled to a uniform size per task type

    Returns:
        Correlation matrix

    """
    new_H = {}
    coords = np.zeros((0, 2))


    for trial in models:
        H = models[trial]['H']
        s = scaled[trial]['Stimulus']

        li = np.asarray(dfs[trial]['Component'])
        array = np.asarray(dfs[trial]['Cumulative variance explained'])
        for j in range(len(array)):
            if array[j] > .8:
                num = j
                break

        H_new = np.zeros((0, H.shape[1]))

        for k in range(num):
            H_new = np.vstack((H_new, H[li[k], :]))

        new_H[trial] = H_new

        for k in range(num):
            ind = np.argpartition(H_new[k], - int(len(H_new[k]) * .025))[-int(len(H_new[k]) * .025):]

            x_ = s.location[ind, 0]
            y_ = s.location[ind, 1]

            points = np.vstack((x_, y_)).T

            coords = np.vstack((coords, points))

        if trial == 'trial05':
            df = pd.DataFrame(new_H[trial])
            component_ids = [str(trial_titles[trial]) + '.1', str(trial_titles[trial]) + '.2', \
                             str(trial_titles[trial]) + '.3', \
                             str(trial_titles[trial]) + '.4', str(trial_titles[trial]) + '.5', \
                             str(trial_titles[trial]) + '.6', str(trial_titles[trial]) + '.7']
            df.index = component_ids[:num]
            df_ = df
        else:

            df = pd.DataFrame(new_H[trial])
            component_ids = [str(trial_titles[trial]) + '.1', str(trial_titles[trial]) + '.2', \
                             str(trial_titles[trial]) + '.3', \
                             str(trial_titles[trial]) + '.4', str(trial_titles[trial]) + '.5', \
                             str(trial_titles[trial]) + '.6', str(trial_titles[trial]) + '.7']
            df.index = component_ids[:num]
            df_ = df_.append(df)

    plt.figure(figsize=(15, 15), dpi=100)
    sns.heatmap(cluster_corr(df_.T.corr()), cmap='cubehelix', square=True, vmin=-.25, vmax=1., cbar=True)
    plt.savefig('../individual_figures/sorted_component_correlation_matrix.png', dpi=100)



def proportion_time_CoP_in_contact_area(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates):

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
    standing = all_CoP_coordinates[all_CoP_coordinates['Trial type'] == 'Standing']
    jumping = all_CoP_coordinates[all_CoP_coordinates['Trial type'] == 'Jumping']


    ### locomotion trials

    step_CoP_coordinates['X_distance'] = np.abs((step_CoP_coordinates['X_contact'] - step_CoP_coordinates['X_pressure'])) / (x_outline.max() - x_outline.min()) * 100
    step_CoP_coordinates['Y_distance'] = np.abs((step_CoP_coordinates['Y_contact'] - step_CoP_coordinates['Y_pressure'])) / (y_outline.max() - y_outline.min()) * 100

    step_CoP_coordinates = step_CoP_coordinates.reset_index()
    step_all_metrics = step_all_metrics.reset_index()

    forces = step_all_metrics['Force'].copy()
    step_CoP_coordinates['Force'] = forces.values

    locomotion = step_CoP_coordinates[step_CoP_coordinates['Force'] > 5.]

    locomotion['CoP inside contact'] = locomotion['CoP inside contact'].replace(
        {1.0: 'Inside', 0.0: 'Outside'})
    count_walk = (locomotion['CoP inside contact'].value_counts() / locomotion[
        'CoP inside contact'].count()) * 100
    count_walk_df = pd.DataFrame(count_walk).rename(columns={'CoP inside contact': 'Locomotion'})

    standing['CoP inside contact'] = standing['CoP inside contact'].replace(
        {1.0: 'Inside', 0.0: 'Outside'})
    stand_walk = (standing['CoP inside contact'].value_counts() / standing[
        'CoP inside contact'].count()) * 100
    count_stand_df = pd.DataFrame(stand_walk).rename(columns={'CoP inside contact': 'Standing'})

    jumping['CoP inside contact'] = jumping['CoP inside contact'].replace(
        {1.0: 'Inside', 0.0: 'Outside'})
    jump_walk = (jumping['CoP inside contact'].value_counts() / jumping[
        'CoP inside contact'].count()) * 100
    count_jump_df = pd.DataFrame(jump_walk).rename(columns={'CoP inside contact': 'Jumping'})

    count_df = pd.concat([count_stand_df, count_walk_df, count_jump_df], axis='columns')

    print(count_df)
    #regions = ['Inside', 'Outside']
    #count_df = count_df.loc[regions]

    plt.figure(figsize=(10, 80), dpi=100)
    count_df.T.plot(kind='bar', stacked=True, color=['green', 'red'],
                    legend=False)
    plt.ylabel('Percentage of time', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    sns.despine()
    plt.savefig("../individual_figures/CoP_in_contact_stacked_bar.png", dpi=100, bbox_inches="tight")

    stand_trials = pd.unique(standing['Trial title'])
    for i in range(len(stand_trials)):

        trialdf = standing[standing['Trial title'] == stand_trials[i]]
        trialdf['CoP inside contact'] = trialdf['CoP inside contact'].replace(
            {1.0: 'Inside', 0.0: 'Outside'})
        count = (trialdf['CoP inside contact'].value_counts() / trialdf[
            'CoP inside contact'].count()) * 100
        countdf = pd.DataFrame(count).rename(columns={'CoP inside contact': stand_trials[i]})

        if i == 0:
            counting_df = countdf
        else:
            counting_df = pd.concat([counting_df, countdf], axis='columns')



    loco_trials = pd.unique(locomotion['Trial title'])
    for i in range(len(loco_trials)):
        trialdf = locomotion[locomotion['Trial title'] == loco_trials[i]]
        trialdf['CoP inside contact'] = trialdf['CoP inside contact'].replace(
            {1.0: 'Inside', 0.0: 'Outside'})
        count = (trialdf['CoP inside contact'].value_counts() / trialdf[
            'CoP inside contact'].count()) * 100
        countdf = pd.DataFrame(count).rename(columns={'CoP inside contact': loco_trials[i]})

        counting_df = pd.concat([counting_df, countdf], axis='columns')



    jump_trials = pd.unique(jumping['Trial title'])
    for i in range(len(jump_trials)):
        trialdf = jumping[jumping['Trial title'] == jump_trials[i]]
        trialdf['CoP inside contact'] = trialdf['CoP inside contact'].replace(
            {1.0: 'Inside', 0.0: 'Outside'})
        count = (trialdf['CoP inside contact'].value_counts() / trialdf[
            'CoP inside contact'].count()) * 100
        countdf = pd.DataFrame(count).rename(columns={'CoP inside contact': jump_trials[i]})

        counting_df = pd.concat([counting_df, countdf], axis='columns')


    plt.figure(figsize=(50, 80), dpi=100)
    counting_df.T.plot(kind='bar', stacked=True, color=['green', 'red'],
                    legend=False)
    plt.ylabel('Percentage of time', fontsize=40)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=35)
    sns.despine()
    plt.savefig("../individual_figures/CoP_in_contact_stacked_bar_all_tasks.png", dpi=100, bbox_inches="tight")