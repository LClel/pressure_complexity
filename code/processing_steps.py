from insole import *
import pickle as pk
from project_insole_constants import *
from project_insole_analysis_information import *
from step_processing_functions import *
from insole_validation import *


def step_data_processing(stomps, idxs):
    filepath_prefix = '../preprocessed_data/'

    ## Normalize step length to 100 time points - removing turns
    normalized_steps = normalize_step_length_across_participants_with_exclusions(filepath_prefix, stomps)

    compressed_pickle('../processed_data/normalized_steps', normalized_steps)

    normalized_steps = decompress_pickle('../processed_data/normalized_steps.pbz2')
    walking_cop_df_creation(normalized_steps, idxs)

    #normalized_slopes = uphill_downhill_normalized_steps(calibration_constants, filepath_prefix)

    #normalized_stairs = upstairs_downstairs_normalized_steps(calibration_constants, filepath_prefix)

    all_metrics = all_metrics_steps(normalized_steps, idxs)

    compressed_pickle('../processed_data/all_metrics_steps', all_metrics)

    regions_all_metrics_steps(normalized_steps, idxs)

    #raw_pressure_steps(normalized_steps, idxs)
