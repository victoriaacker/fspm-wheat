# -*- coding: latin-1 -*-

from __future__ import print_function

import os
import sys
import getopt

import pandas as pd
import ast

from fspmwheat import fspmwheat_postprocessing
from example.Scenarios_Climate import main
from example.Scenarios_Climate import tools
# from example.Scenarios_monoculms import additional_graphs


def run_fspmwheat(scenario_id=1, inputs_dirpath='inputs', outputs_dir_path='outputs'):
    """
    Run the main.py of fspmwheat using data from a specific scenario

    :param int scenario_id: the index of the scenario to be read in the CSV file containing the list of scenarios
    :param str inputs_dirpath: the path directory of inputs
    :param str outputs_dir_path: the path to save outputs
    """

    # Scenario to be run
    scenarios_df = pd.read_csv(os.path.join(inputs_dirpath, 'scenarios_list.csv'), index_col='Scenario')
    scenarios_df['drought_trigger'].fillna(False, inplace=True)
    scenario_conditions = scenarios_df.loc[scenario_id].to_dict()
    scenario_name = scenario_conditions['Scenario_label']


    # -- SIMULATION PARAMETERS --

    # Create dict of parameters for the scenario
    scenario_parameters = tools.buildDic(scenario_conditions)

    # Do run the simulation?
    RUN_SIMU = scenario_parameters.get('Run_Simulation', True)

    SIMULATION_LENGTH = scenario_parameters.get('Simulation_Length', 3000)

    # Do run the postprocessing?
    RUN_POSTPROCESSING = scenario_parameters.get('Run_Postprocessing', True)  #: TODO separate postprocessings coming from other models

    # Do generate the graphs?
    GENERATE_GRAPHS = scenario_parameters.get('Generate_Graphs', False)  #: TODO separate postprocessings coming from other models

    # Inputs of the scenario
    scenario_meteo = scenario_parameters.get('METEO_FILENAME', 'meteo_CO2_400.csv')

    # Build N Fertilizations dict
    N_FERTILIZATIONS = {}
    if 'constant_Conc_Nitrates' in scenario_parameters:
        N_FERTILIZATIONS = {'constant_Conc_Nitrates': scenario_parameters.get('constant_Conc_Nitrates')}
    # Setup N_fertilizations time if time interval is given:
    if 'Fertilization' in scenario_parameters:
        N_FERTILIZATIONS = ast.literal_eval(scenario_parameters['Fertilization'])

    # Tiller
    TILLERS = {'T1': 0.5, 'T2': 0.5, 'T3': 0.5, 'T4': 0.5}
    if 'Tillers' in scenario_parameters:
        TILLERS = ast.literal_eval(scenario_parameters['Tillers'])

    # Drought
    drought_trigger = scenario_parameters.get('drought_trigger', 'False')
    if drought_trigger != 'False':
        drought_trigger = float(drought_trigger)
    else:
        drought_trigger = False
    stop_drought_SRWC = scenario_parameters.get('stop_drought_SRWC', 'False')


    if RUN_SIMU or RUN_POSTPROCESSING or GENERATE_GRAPHS:

        # -- SIMULATION DIRECTORIES --

        # Path of the directory which contains the outputs of the model
        scenario_dirpath = os.path.join(outputs_dir_path, scenario_name)

        # Create the directory of the Scenario where results will be stored
        if not os.path.exists(scenario_dirpath):
            os.mkdir(scenario_dirpath)

        # Create directory paths for graphs, outputs and postprocessings of this scenario
        scenario_graphs_dirpath = os.path.join(scenario_dirpath, 'graphs')
        if not os.path.exists(scenario_graphs_dirpath):
            os.mkdir(scenario_graphs_dirpath)

        # Outputs
        scenario_outputs_dirpath = os.path.join(scenario_dirpath, 'outputs')
        if not os.path.exists(scenario_outputs_dirpath):
            os.mkdir(scenario_outputs_dirpath)

        # Postprocessings
        scenario_postprocessing_dirpath = os.path.join(scenario_dirpath, 'postprocessing')
        if not os.path.exists(scenario_postprocessing_dirpath):
            os.mkdir(scenario_postprocessing_dirpath)

        # -- RUN main fspmwheat --
        print(scenario_name)
        try:
            main.main(simulation_length=SIMULATION_LENGTH,
                      run_simu=RUN_SIMU, run_postprocessing=RUN_POSTPROCESSING, generate_graphs=GENERATE_GRAPHS, run_from_outputs=False, forced_start_time=1176,
                      METEO_FILENAME=scenario_meteo,
                      N_fertilizations=N_FERTILIZATIONS,
                      PLANT_DENSITY={1: 250},
                      GRAPHS_DIRPATH=scenario_graphs_dirpath,
                      INPUTS_DIRPATH=inputs_dirpath,
                      OUTPUTS_DIRPATH=scenario_outputs_dirpath,
                      POSTPROCESSING_DIRPATH=scenario_postprocessing_dirpath,
                      update_parameters_all_models=scenario_parameters,
                      tillers_replications=TILLERS,
                      drought_trigger=drought_trigger, stop_drought_SRWC=stop_drought_SRWC)
            # if GENERATE_GRAPHS:
            #     additional_graphs.graph_summary(scenario_id, scenario_graphs_dirpath,
            #                                     graph_list=['LAI', 'sum_dry_mass_axis', 'shoot_roots_ratio_axis', 'N_content_shoot_axis', 'Conc_Amino_acids_phloem', 'Conc_Sucrose_phloem', 'leaf_Lmax',
            #                                                 'green_area_blade'])
            # if RUN_POSTPROCESSING:
            #     fspmwheat_postprocessing.leaf_traits(scenario_outputs_dirpath, scenario_postprocessing_dirpath)
            #     fspmwheat_postprocessing.table_C_usages(scenario_postprocessing_dirpath)
            #     fspmwheat_postprocessing.calculate_performance_indices(scenario_outputs_dirpath, scenario_postprocessing_dirpath, os.path.join(INPUTS_DIRPATH, scenario.get('METEO_FILENAME')),
            #                                                            scenario.get('Plant_Density', 250.))
            #     fspmwheat_postprocessing.canopy_dynamics(scenario_postprocessing_dirpath, os.path.join(INPUTS_DIRPATH, scenario.get('METEO_FILENAME')),
            #                                              scenario.get('Plant_Density', 250.))

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    scenario = 3
    inputs = 'inputs'
    outputs = 'outputs'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:s:d", ["inputs=", "outputs=", "scenario="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--inputs"):
            inputs = arg
        elif opt in ("-o", "--outputs"):
            outputs = arg
        elif opt in ("-s", "--scenario"):
            scenario = int(arg)

    run_fspmwheat(inputs_dirpath=inputs, outputs_dir_path=outputs, scenario_id=scenario)
