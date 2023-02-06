from data_collation import *
from figures_functions import *
from processing_all_data import *
from processing_steps import *
from insole import *
import pickle as pk
from run_nmf import *
from scale_data import *
from figures import *
from project_insole_statistics import *
""" This file runs the full pipeline that includes data preprocessing, analysis and figure generation.
    Documentation on individual functions can be found within their respective files.
"""

stomps = decompress_pickle('../processed_data/pressure_stomps.pbz2')

data_collation_function()

idxs = decompress_pickle('../processed_data/participant_stimulus_indexes.pbz2')

calibration_constants = decompress_pickle('../processed_data/calibration_constants.pbz2')

step_data_processing(stomps, idxs)

all_data_processing(stomps, idxs, calibration_constants)

scale_all_data(stomps)

scale_data_trial_type(stomps)

scale_data_trials(stomps)

scaled = decompress_pickle('../scaled_data/scaled raw data by trial.pbz2')

nmf_all_trials(scaled)

#scaled_trial_type = decompress_pickle('../scaled_data/scaled raw data by trial type.pbz2')

#nmf_trial_type(scaled_trial_type)

#scaled = decompress_pickle('/Users/lukecleland/Documents/PhD/Research projects/Project insole/project_insole/scaled_data/scaled raw data by trial.pbz2')
#models = decompress_pickle('/Users/lukecleland/Documents/PhD/Research projects/Project insole/project_insole/processed_data/nmf_weights.pbz2')
#dfs = decompress_pickle('/Users/lukecleland/Documents/PhD/Research projects/Project insole/project_insole/processed_data/nmf_dfs.pbz2')
#inverse_transform_trials(dfs, models, scaled)

#trial_type_dfs = decompress_pickle('/Users/lukecleland/Documents/PhD/Research projects/Project insole/project_insole/processed_data/trial_type_nmf_dfs.pbz2')
#trial_type_models = decompress_pickle('/Users/lukecleland/Documents/PhD/Research projects/Project insole/project_insole/processed_data/trial_type_nmf_weights.pbz2')
#inverse_transform_trial_type(trial_type_dfs, trial_type_models, scaled)

insole_figures(stomps)