from insole import *
import pickle as pk
from project_insole_constants import *
from project_insole_analysis_information import *
from step_processing_functions import *
from all_data_processing_functions import *
from insole_validation import *


def all_data_processing(stomps, idxs, calibration_constants):

    all_data = all_metrics(stomps)

    compressed_pickle('../processed_data/all_metrics_df', all_data)

    regions_all_metrics(idxs, stomps)

    cop_df_creation(stomps)

    raw_pressure_all(idxs, stomps)

    time_each_foot_in_contact_with_ground(filepath_prefix)

    proportion_mapped_sensors()

    project_insole_calibration(separation_indexes, calibration_constants)