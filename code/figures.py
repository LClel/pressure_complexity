from project_insole_constants import *
from project_insole_analysis_information import *
from step_processing_functions import *
from all_data_processing_functions import *
from figures_functions import *
from run_nmf import *
from insole_validation import *
import pickle as pk
from project_insole_statistics import *

def insole_figures(stomps):
    """ Generates all figures - documentation of what each figure generates is within paper_figures_functions.py

    Args:
        stomps:

    Returns:

    """
    all_metrics = decompress_pickle('../processed_data/all_metrics_df.pbz2')

    step_all_metrics = decompress_pickle('../processed_data/all_metrics_steps.pbz2')

    compare_mean_area(all_metrics, step_all_metrics)

    calibration_constants = decompress_pickle('../processed_data/calibration_constants.pbz2')

    pressure_alignment_on_foot()

    df = project_insole_calibration(separation_indexes, calibration_constants)
    compressed_pickle('../processed_data/calibration_dfs', df)

    normalized_steps = decompress_pickle('../processed_data/normalized_steps.pbz2')

    pressure_overview(all_metrics, step_all_metrics)

    max_per_trial_plot(all_metrics, step_all_metrics)

    mean_per_trial_plot(step_all_metrics, all_metrics)

    mean_per_trial_plot_area(step_all_metrics, all_metrics)

    contact_area_overview(all_metrics, step_all_metrics)

    force_over_time(step_all_metrics)

    area_over_time(step_all_metrics)

    all_CoPs = walking_trial_CoPs(normalized_steps)
    walking_CoP_variation_plots_on_foot(all_CoPs)

    standing_CoPs = standing_trial_CoPs(stomps)
    standing_trial_CoP_variation_plots(standing_CoPs)

    jumping_CoPs = jumping_trial_CoPs(stomps)
    jumping_trial_CoP_variation_plots(jumping_CoPs)


    CoC_traces(normalized_steps)

    CoC_trace_colorbar()

    CoP_traces(normalized_steps)

    CoP_stacked_bars(all_metrics, step_all_metrics)

    all_CoP_coordinates = pd.read_csv('../processed_data/all_CoP_coordinates_df.csv')
    step_CoP_coordinates = pd.read_csv('../processed_data/walking_CoP_coordinates_df.csv')

    CoP_CoC_correlation(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates)

    CoP_CoC_distance_bars(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates)

    correlation_between_CoP_CoC(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates)

    CoP_CoC_scatterplot(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates)

    pressure_area_correlation_plots(step_all_metrics, all_metrics)

    force_area_correlation(step_all_metrics, all_metrics)

    pressure_area_combinations(all_metrics, step_all_metrics)

    CoP_density(all_CoP_coordinates, step_CoP_coordinates)

    correlation_between_CoP_CoC(all_metrics, step_all_metrics, all_CoP_coordinates, step_CoP_coordinates)

    dfs = decompress_pickle('../processed_data/nmf_dfs.pbz2')
    dfs = sort_nmf_df(dfs)

    models = decompress_pickle('../processed_data/nmf_weights.pbz2')

    scaled = decompress_pickle('../scaled_data/scaled raw data by trial.pbz2')

    nmf_scree_plot(dfs)

    nmf_components_required_on_foot(dfs, models, scaled)

    nmf_walking_hotspots(dfs, models, scaled)

    NMF_component_clusters(dfs, models, scaled)

    component_correlation_matrix(dfs, models, scaled)

    trial_type_dfs = decompress_pickle('../processed_data/trial_type_nmf_dfs.pbz2')
    trial_type_dfs = sort_nmf_df(trial_type_dfs)

    trial_type_nmf_scree_plot(trial_type_dfs)

    trial_type_models = decompress_pickle('../processed_data/trial_type_nmf_weights.pbz2')

    inverse_transform_trial_type(trial_type_dfs, trial_type_models, scaled)

    trial_type_scaled = decompress_pickle('../scaled_data/scaled raw data by trial type.pbz2')
    trial_type_component_clusters(trial_type_dfs, trial_type_models, trial_type_scaled)

    trial_type_nmf_components_required_on_foot(trial_type_dfs, trial_type_models, trial_type_scaled)

    contact_area_probability(scaled)

    individual_frame_contact(scaled)

    scaled_all = decompress_pickle('../scaled_data/scaled raw data.pbz2')
    contact_area_probability_all_data(scaled_all)
