# -*- coding: latin-1 -*-
from math import pi
import math
import numpy as np

from turgorgrowth import model as turgorgrowth_model, simulation as turgorgrowth_simulation, \
    converter as turgorgrowth_converter, postprocessing as turgorgrowth_postprocessing

from fspmwheat import tools

"""
    fspmwheat.turgorgrowth_facade
    ~~~~~~~~~~~~~~~~~~~~~~~~

    The module :mod:`fspmwheat.turgorgrowth_facade` is a facade of the model Turgor-Growth.

    This module permits to initialize and run the model Turgor-Growth from a :class:`MTG <openalea.mtg.mtg.MTG>`
    in a convenient and transparent way, wrapping all the internal complexity of the model, and dealing
    with all the tedious initialization and conversion processes.

    :license: TODO, see LICENSE for details.

"""

#: the mapping of Turgor-Growth organ classes to the attributes in axis and phytomer which represent an organ
TURGORGROWTH_ATTRIBUTES_MAPPING = {turgorgrowth_model.Internode: 'internode', turgorgrowth_model.Lamina: 'lamina', turgorgrowth_model.Sheath: 'sheath',
                                   turgorgrowth_model.Roots: 'roots', turgorgrowth_model.HiddenZone: 'hiddenzone', turgorgrowth_model.Xylem: 'xylem'}

#: the mapping of organs (which belong to an axis) labels in MTG to organ classes in Turgor-Growth
MTG_TO_TURGORGROWTH_AXES_ORGANS_MAPPING = {'xylem': turgorgrowth_model.Xylem, 'roots': turgorgrowth_model.Roots}

#: the mapping of organs (which belong to a phytomer) labels in MTG to organ classes in Turgor-Growth
MTG_TO_TURGORGROWTH_PHYTOMERS_ORGANS_MAPPING = {'internode': turgorgrowth_model.Internode, 'blade': turgorgrowth_model.Lamina, 'sheath': turgorgrowth_model.Sheath,
                                                'hiddenzone': turgorgrowth_model.HiddenZone}

#: the mapping of Turgor-Growth photosynthetic organs to Turgor-Growth photosynthetic organ elements
TURGORGROWTH_ORGANS_TO_ELEMENTS_MAPPING = {turgorgrowth_model.Internode: turgorgrowth_model.InternodeElement, turgorgrowth_model.Lamina: turgorgrowth_model.LaminaElement,
                                           turgorgrowth_model.Sheath: turgorgrowth_model.SheathElement}

#: the parameters and variables which define the state of a Turgor-Growth population
POPULATION_STATE_VARIABLE = set(turgorgrowth_simulation.Simulation.PLANTS_STATE + turgorgrowth_simulation.Simulation.AXES_STATE +
                                turgorgrowth_simulation.Simulation.ORGANS_STATE + turgorgrowth_simulation.Simulation.PHYTOMERS_STATE +
                                turgorgrowth_simulation.Simulation.HIDDENZONE_STATE + turgorgrowth_simulation.Simulation.ELEMENTS_STATE)

#: all the variables of a Turgor-Growth population computed during a run step of the simulation
POPULATION_RUN_VARIABLES = set(turgorgrowth_simulation.Simulation.PLANTS_RUN_VARIABLES + turgorgrowth_simulation.Simulation.AXES_RUN_VARIABLES +
                               turgorgrowth_simulation.Simulation.PHYTOMERS_RUN_VARIABLES + turgorgrowth_simulation.Simulation.ORGANS_RUN_VARIABLES +
                               turgorgrowth_simulation.Simulation.HIDDENZONE_RUN_VARIABLES + turgorgrowth_simulation.Simulation.ELEMENTS_RUN_VARIABLES)

#: all the variables to be stored in the MTG
MTG_RUN_VARIABLES = set(list(POPULATION_RUN_VARIABLES) + turgorgrowth_simulation.Simulation.SOIL_RUN_VARIABLES)

# number of seconds in 1 hour
HOUR_TO_SECOND_CONVERSION_FACTOR = 3600


class TurgorGrowthFacade(object):
    """
    The TurgorGrowthFacade class permits to initialize, run the model Turgor-Growth
    from a :class:`MTG <openalea.mtg.mtg.MTG>`, and update the MTG and the dataframes
    shared between all models.

    Use :meth:`run` to run the model.

    """

    def __init__(self, shared_mtg, delta_t,
                 model_axes_inputs_df,
                 model_hiddenzones_inputs_df,
                 model_elements_inputs_df,
                 model_organs_inputs_df,
                 model_soil_inputs_df,
                 shared_axes_inputs_outputs_df,
                 shared_hiddenzones_inputs_outputs_df,
                 shared_elements_inputs_outputs_df,
                 shared_organs_inputs_outputs_df,
                 shared_soil_inputs_outputs_df,
                 update_shared_df=True):
        """
                :param openalea.mtg.mtg.MTG shared_mtg: The MTG shared between all models.
                :param int delta_t: The delta between two runs, in seconds.
                :param pandas.DataFrame model_hiddenzones_inputs_df: the inputs of the model at hiddenzones scale.
                :param pandas.DataFrame model_elements_inputs_df: the inputs of the model at elements scale.
                :param pandas.DataFrame shared_hiddenzones_inputs_outputs_df: the dataframe of inputs and outputs at hiddenzones scale shared between all models.
                :param pandas.DataFrame shared_elements_inputs_outputs_df: the dataframe of inputs and outputs at elements scale shared between all models.
                :param bool update_shared_df: If `True`  update the shared dataframes at init and at each run (unless stated otherwise)
        """
        
        self._shared_mtg = shared_mtg  #: the MTG shared between all models

        self._simulation = turgorgrowth_simulation.Simulation(delta_t=delta_t)

        self.population, mapping_topology = turgorgrowth_converter.from_dataframes(model_hiddenzones_inputs_df, model_elements_inputs_df, model_organs_inputs_df, model_soil_inputs_df)

        self._simulation.initialize(self.population, mapping_topology)

        self._update_shared_MTG()

        self._shared_axes_inputs_outputs_df = shared_axes_inputs_outputs_df         #: the dataframe at axes scale shared between all models
        self._shared_hiddenzones_inputs_outputs_df = shared_hiddenzones_inputs_outputs_df         #: the dataframe at hiddenzones scale shared between all models
        self._shared_elements_inputs_outputs_df = shared_elements_inputs_outputs_df               #: the dataframe at elements scale shared between all models
        self._shared_organs_inputs_outputs_df = shared_organs_inputs_outputs_df
        self._shared_soil_inputs_outputs_df = shared_soil_inputs_outputs_df
        self._update_shared_df = update_shared_df
        if self._update_shared_df:
            self._update_shared_dataframes(turgorgrowth_axes_data_df=model_axes_inputs_df,
                                           turgorgrowth_hiddenzones_data_df=model_hiddenzones_inputs_df,
                                           turgorgrowth_elements_data_df=model_elements_inputs_df,
                                           turgorgrowth_organs_data_df=model_organs_inputs_df,
                                           turgorgrowth_soil_data_df=model_soil_inputs_df)

    # def run(self, SRWC=80, update_shared_df=False):
    def run(self, SRWC, sucrose, amino_acids, proteins, update_shared_df=False):

        """
        Run the model and update the MTG and the dataframes shared between all models.
        """
        self._initialize_model(SRWC=SRWC, sucrose=sucrose, amino_acids=amino_acids, proteins=proteins)
        self._simulation.run()
        self._update_shared_MTG()

        if update_shared_df or (update_shared_df is None and self._update_shared_df):
            turgorgrowth_axes_inputs_outputs_df, turgorgrowth_hiddenzones_inputs_outputs_df, turgorgrowth_elements_inputs_outputs_df, turgorgrowth_organs_inputs_outputs_df, turgorgrowth_soil_inputs_outputs_df = turgorgrowth_converter.to_dataframes(self._simulation.population)

            self._update_shared_dataframes(turgorgrowth_axes_data_df=turgorgrowth_axes_inputs_outputs_df,
                                           turgorgrowth_hiddenzones_data_df=turgorgrowth_hiddenzones_inputs_outputs_df,
                                           turgorgrowth_elements_data_df=turgorgrowth_elements_inputs_outputs_df,
                                           turgorgrowth_organs_data_df=turgorgrowth_organs_inputs_outputs_df,
                                           turgorgrowth_soil_data_df=turgorgrowth_soil_inputs_outputs_df)

    # @staticmethod
    # def postprocessing(axes_outputs_df, hiddenzone_outputs_df, elements_outputs_df, organs_outputs_df, soil_outputs_df, delta_t):
    #     """
    #     Run the postprocessing.
    #     """
    #     (axes_postprocessing_df, hiddenzones_postprocessing_df, elements_postprocessing_df, organs_postprocessing_df, soil_postprocessing_df) = turgorgrowth_postprocessing.postprocessing(axes_df=axes_outputs_df, hiddenzones_df=hiddenzone_outputs_df, elements_df=elements_outputs_df, organs_df=organs_outputs_df, soil_df=soil_outputs_df, delta_t=delta_t)
    #     return axes_postprocessing_df, hiddenzones_postprocessing_df, elements_postprocessing_df, organs_postprocessing_df, soil_postprocessing_df

    @staticmethod
    def postprocessing(axes_outputs_df, hiddenzone_outputs_df, elements_outputs_df, organs_outputs_df,  delta_t):
        """
        Run the postprocessing.
        """
        (axes_postprocessing_df, hiddenzones_postprocessing_df, elements_postprocessing_df, organs_postprocessing_df) = turgorgrowth_postprocessing.postprocessing(axes_df=axes_outputs_df, hiddenzones_df=hiddenzone_outputs_df, elements_df=elements_outputs_df, organs_df=organs_outputs_df, delta_t=delta_t)
        return axes_postprocessing_df, hiddenzones_postprocessing_df, elements_postprocessing_df, organs_postprocessing_df

    # @staticmethod
    # def graphs(axes_postprocessing_df, hiddenzones_postprocessing_df, elements_postprocessing_df, organs_postprocessing_df, soils_postprocessing_df, graphs_dirpath='.'):
    #     """
    #     Generate the graphs and save them into `graphs_dirpath`.
    #     """
    #     turgorgrowth_postprocessing.generate_graphs(axes_df=axes_postprocessing_df, hiddenzones_df=hiddenzones_postprocessing_df, elements_df=elements_postprocessing_df, organs_df=organs_postprocessing_df, soil_df=soils_postprocessing_df, graphs_dirpath=graphs_dirpath)

    @staticmethod
    def graphs(axes_postprocessing_df, hiddenzones_postprocessing_df, elements_postprocessing_df, organs_postprocessing_df, graphs_dirpath='.'):
        """
        Generate the graphs and save them into `graphs_dirpath`.
        """
        turgorgrowth_postprocessing.generate_graphs(axes_df=axes_postprocessing_df, hiddenzones_df=hiddenzones_postprocessing_df, elements_df=elements_postprocessing_df, organs_df=organs_postprocessing_df, graphs_dirpath=graphs_dirpath)

    def _initialize_model(self, SRWC=80, sucrose=sucrose, amino_acids=amino_acids, proteins=proteins):
        """
        Initialize the inputs of the model from the MTG shared between all models and the soils.
        """

        self.population = turgorgrowth_model.Population()
        mapping_topology = {'predecessor': {}, 'successor': {}}

        # traverse the MTG recursively from top

        for mtg_plant_vid in self._shared_mtg.components_iter(self._shared_mtg.root):
            mtg_plant_index = int(self._shared_mtg.index(mtg_plant_vid))

            # create a new plant
            turgorgrowth_plant = turgorgrowth_model.Plant(mtg_plant_index)
            is_valid_plant = False
            for mtg_axis_vid in self._shared_mtg.components_iter(mtg_plant_vid):
                mtg_axis_label = self._shared_mtg.label(mtg_axis_vid)
                #: keep only MS TODO: temporary
                if mtg_axis_label != 'MS':
                    continue

                #: MS
                # create a new axis
                turgorgrowth_axis = turgorgrowth_model.Axis(mtg_axis_label)
                is_valid_axis = True
                for turgorgrowth_organ_class in (turgorgrowth_model.Roots, turgorgrowth_model.Xylem):
                    mtg_organ_label = turgorgrowth_converter.TURGORGROWTH_CLASSES_TO_DATAFRAME_ORGANS_MAPPING[turgorgrowth_organ_class]
                    # create a new organ
                    turgorgrowth_organ = turgorgrowth_organ_class(mtg_organ_label)
                    mtg_axis_properties = self._shared_mtg.get_vertex_property(mtg_axis_vid)
                    if mtg_organ_label in mtg_axis_properties:
                        mtg_organ_properties = mtg_axis_properties[mtg_organ_label]
                        turgorgrowth_organ_data_names = set(turgorgrowth_simulation.Simulation.ORGANS_STATE).intersection(turgorgrowth_organ.__dict__)
                        if set(mtg_organ_properties).issuperset(turgorgrowth_organ_data_names):
                            turgorgrowth_organ_data_dict = {}
                            for turgorgrowth_organ_data_name in turgorgrowth_organ_data_names:
                                turgorgrowth_organ_data_dict[turgorgrowth_organ_data_name] = mtg_organ_properties[turgorgrowth_organ_data_name]

                                # Debug: Tell if missing input variable
                                if math.isnan(mtg_organ_properties[turgorgrowth_organ_data_name]) or mtg_organ_properties[turgorgrowth_organ_data_name] is None:
                                    print('Missing variable', turgorgrowth_organ_data_name, 'for vertex id', mtg_axis_vid, 'which is', mtg_organ_label)

                            turgorgrowth_organ.__dict__.update(turgorgrowth_organ_data_dict)

                            # Update parameters if specified
                            # if mtg_organ_label in self._update_parameters:
                            #     turgorgrowth_organ.PARAMETERS.__dict__.update(self._update_parameters[mtg_organ_label])

                            turgorgrowth_organ.initialize()
                            # add the new organ to current axis
                            setattr(turgorgrowth_axis, mtg_organ_label, turgorgrowth_organ)
                if not is_valid_axis:
                    continue

                has_valid_phytomer = False
                for mtg_metamer_vid in self._shared_mtg.components_iter(mtg_axis_vid):
                    mtg_metamer_index = int(self._shared_mtg.index(mtg_metamer_vid))

                    # create a new phytomer
                    turgorgrowth_phytomer = turgorgrowth_model.Phytomer(mtg_metamer_index)
                    mtg_hiddenzone_label = turgorgrowth_converter.TURGORGROWTH_CLASSES_TO_DATAFRAME_ORGANS_MAPPING[turgorgrowth_model.HiddenZone]
                    mtg_metamer_properties = self._shared_mtg.get_vertex_property(mtg_metamer_vid)

                    if mtg_hiddenzone_label in mtg_metamer_properties and mtg_metamer_properties[mtg_hiddenzone_label]['leaf_is_growing']:
                        has_valid_hiddenzone = True
                        turgorgrowth_hiddenzone = turgorgrowth_model.HiddenZone(label=mtg_hiddenzone_label)
                        mtg_hiddenzone_properties = mtg_metamer_properties[mtg_hiddenzone_label]
                        # turgorgrowth_hiddenzone_data_names = set(turgorgrowth_simulation.Simulation.HIDDENZONE_STATE).intersection(turgorgrowth_hiddenzone.__dict__)
                        # Adding aggregated variables into inputs
                        # turgorgrowth_hiddenzone_data_names = set(turgorgrowth_simulation.Simulation.HIDDENZONE_STATE + turgorgrowth_simulation.Simulation.HIDDENZONE_FLUXES).intersection(turgorgrowth_hiddenzone.__dict__)
                        turgorgrowth_hiddenzone_data_names = set(turgorgrowth_simulation.Simulation.HIDDENZONE_RUN_VARIABLES).intersection(turgorgrowth_hiddenzone.__dict__)
                        if mtg_hiddenzone_properties.get('leaf_pseudo_age') == 0:  # First time hiddenzone passes into turgorgrowth model
                            missing_initial_hiddenzone_properties = turgorgrowth_hiddenzone_data_names - set(mtg_hiddenzone_properties)
                            turgorgrowth_hiddenzone_data_names -= missing_initial_hiddenzone_properties

                        if set(mtg_hiddenzone_properties).issuperset(turgorgrowth_hiddenzone_data_names):
                            turgorgrowth_hiddenzone_data_dict = {}
                            for turgorgrowth_hiddenzone_data_name in turgorgrowth_hiddenzone_data_names:
                                mtg_hiddenzone_data_value = mtg_hiddenzone_properties.get(turgorgrowth_hiddenzone_data_name)
                                turgorgrowth_hiddenzone_data_dict[turgorgrowth_hiddenzone_data_name] = mtg_hiddenzone_data_value
                            turgorgrowth_hiddenzone.__dict__.update(turgorgrowth_hiddenzone_data_dict)
                        # add the new hiddenzone to current phytomer
                        setattr(turgorgrowth_phytomer, mtg_hiddenzone_label, turgorgrowth_hiddenzone)

                    else:
                        has_valid_hiddenzone = False

                    # create a new organ
                    has_valid_organ = False
                    for mtg_organ_vid in self._shared_mtg.components_iter(mtg_metamer_vid):
                        mtg_organ_label = self._shared_mtg.label(mtg_organ_vid)
                        if mtg_organ_label not in MTG_TO_TURGORGROWTH_PHYTOMERS_ORGANS_MAPPING or self._shared_mtg.get_vertex_property(mtg_organ_vid)['length'] == 0:
                            continue
                        turgorgrowth_organ_class = MTG_TO_TURGORGROWTH_PHYTOMERS_ORGANS_MAPPING[mtg_organ_label]
                        turgorgrowth_organ = turgorgrowth_organ_class(mtg_organ_label)

                        # # Update parameters if specified
                        # if 'PhotosyntheticOrgan' in self._update_parameters:
                        #     turgorgrowth_organ.PARAMETERS.__dict__.update(self._update_parameters['PhotosyntheticOrgan'])

                        turgorgrowth_organ.initialize()
                        has_valid_element = False

                        # create a new element
                        for mtg_element_vid in self._shared_mtg.components_iter(mtg_organ_vid):
                            mtg_element_label = self._shared_mtg.label(mtg_element_vid)
                            if mtg_element_label not in turgorgrowth_converter.DATAFRAME_TO_TURGORGROWTH_ELEMENTS_NAMES_MAPPING \
                                    or (self._shared_mtg.get_vertex_property(mtg_element_vid)['length'] == 0) \
                                    or ((mtg_element_label == 'HiddenElement') and (self._shared_mtg.get_vertex_property(mtg_element_vid).get('is_growing', True))):
                                continue
                            has_valid_element = True
                            turgorgrowth_element = TURGORGROWTH_ORGANS_TO_ELEMENTS_MAPPING[turgorgrowth_organ_class](label=mtg_element_label)
                            mtg_element_properties = self._shared_mtg.get_vertex_property(mtg_element_vid)
                            # turgorgrowth_element_data_names = set(turgorgrowth_simulation.Simulation.ELEMENTS_STATE).intersection(turgorgrowth_element.__dict__)
                            # Adding aggregated variables into inputs
                            # turgorgrowth_element_data_names = set(turgorgrowth_simulation.Simulation.ELEMENTS_STATE + turgorgrowth_simulation.Simulation.ELEMENTS_FLUXES).intersection(turgorgrowth_element.__dict__)
                            turgorgrowth_element_data_names = set(turgorgrowth_simulation.Simulation.ELEMENTS_RUN_VARIABLES).intersection(turgorgrowth_element.__dict__)
                            if mtg_element_properties.get('age') == 0:  # First time element passes into turgorgrowth model
                                missing_initial_element_properties = turgorgrowth_element_data_names - set(mtg_element_properties)
                                turgorgrowth_element_data_names -= missing_initial_element_properties

                            if set(mtg_element_properties).issuperset(turgorgrowth_element_data_names):
                                turgorgrowth_element_data_dict = {}
                                for turgorgrowth_element_data_name in turgorgrowth_element_data_names:
                                    mtg_element_data_value = mtg_element_properties.get(turgorgrowth_element_data_name)
                                    turgorgrowth_element_data_dict[turgorgrowth_element_data_name] = mtg_element_data_value
                                turgorgrowth_element.__dict__.update(turgorgrowth_element_data_dict)
                                # add element to organ
                                setattr(turgorgrowth_organ, turgorgrowth_converter.DATAFRAME_TO_TURGORGROWTH_ELEMENTS_NAMES_MAPPING[mtg_element_label], turgorgrowth_element)

                        if has_valid_element:
                            has_valid_organ = True
                            setattr(turgorgrowth_phytomer, TURGORGROWTH_ATTRIBUTES_MAPPING[turgorgrowth_organ_class], turgorgrowth_organ)

                    if has_valid_organ or has_valid_hiddenzone:
                        turgorgrowth_axis.phytomers.append(turgorgrowth_phytomer)
                        has_valid_phytomer = True

                if not has_valid_phytomer:
                    is_valid_axis = False

                if is_valid_axis:
                    turgorgrowth_plant.axes.append(turgorgrowth_axis)
                    is_valid_plant = True

            if is_valid_plant:
                self.population.plants.append(turgorgrowth_plant)

        #self._simulation.initialize(self.population, mapping_topology, SRWC=SRWC)
        self._simulation.initialize(self.population, mapping_topology, SRWC=SRWC, sucrose = sucrose, amino_acids = amino_acids, proteins = proteins)

    def _update_shared_MTG(self):
        """
        Update the MTG shared between all models from the population of Turgor-Growth.
        """
        # add the missing properties
        mtg_property_names = self._shared_mtg.property_names()
        for turgorgrowth_data_name in MTG_RUN_VARIABLES:
            if turgorgrowth_data_name not in mtg_property_names:
                self._shared_mtg.add_property(turgorgrowth_data_name)
        for turgorgrowth_organ_label in list(MTG_TO_TURGORGROWTH_AXES_ORGANS_MAPPING.keys()) + [turgorgrowth_converter.TURGORGROWTH_CLASSES_TO_DATAFRAME_ORGANS_MAPPING[turgorgrowth_model.HiddenZone]]:
            if turgorgrowth_organ_label not in mtg_property_names:
                self._shared_mtg.add_property(turgorgrowth_organ_label)

        mtg_plants_iterator = self._shared_mtg.components_iter(self._shared_mtg.root)
        # traverse Turgor_Growth population from top
        for turgorgrowth_plant in self.population.plants:
            turgorgrowth_plant_index = turgorgrowth_plant.index
            while True:
                mtg_plant_vid = next(mtg_plants_iterator)
                if int(self._shared_mtg.index(mtg_plant_vid)) == turgorgrowth_plant_index:
                    break
            mtg_axes_iterator = self._shared_mtg.components_iter(mtg_plant_vid)
            for turgorgrowth_axis in turgorgrowth_plant.axes:
                turgorgrowth_axis_label = turgorgrowth_axis.label
                while True:
                    mtg_axis_vid = next(mtg_axes_iterator)
                    if self._shared_mtg.label(mtg_axis_vid) == turgorgrowth_axis_label:
                        break

                # : __________________________________________________________________________________________________________________________

                # XYLEM

                turgorgrowth_axis_property_names = [property_name for property_name in turgorgrowth_simulation.Simulation.AXES_RUN_VARIABLES if hasattr(turgorgrowth_axis, property_name)]
                for turgorgrowth_axis_property_name in turgorgrowth_axis_property_names:
                    turgorgrowth_axis_property_value = getattr(turgorgrowth_axis, turgorgrowth_axis_property_name)
                    self._shared_mtg.property(turgorgrowth_axis_property_name)[mtg_axis_vid] = turgorgrowth_axis_property_value

                for mtg_organ_label in MTG_TO_TURGORGROWTH_AXES_ORGANS_MAPPING.keys():
                    if mtg_organ_label not in self._shared_mtg.get_vertex_property(mtg_axis_vid):
                        # Add a property describing the organ to the current axis of the MTG
                        self._shared_mtg.property(mtg_organ_label)[mtg_axis_vid] = {}
                    # Update the property describing the organ of the current axis in the MTG
                    turgorgrowth_organ = getattr(turgorgrowth_axis, mtg_organ_label)
                    mtg_organ_properties = self._shared_mtg.get_vertex_property(mtg_axis_vid)[mtg_organ_label]
                    for turgorgrowth_property_name in turgorgrowth_simulation.Simulation.ORGANS_RUN_VARIABLES:
                        if hasattr(turgorgrowth_organ, turgorgrowth_property_name):
                            mtg_organ_properties[turgorgrowth_property_name] = getattr(turgorgrowth_organ, turgorgrowth_property_name)
                mtg_metamers_iterator = self._shared_mtg.components_iter(mtg_axis_vid)

                # : __________________________________________________________________________________________________________________________

                for turgorgrowth_phytomer in turgorgrowth_axis.phytomers:
                    turgorgrowth_phytomer_index = turgorgrowth_phytomer.index
                    while True:
                        mtg_metamer_vid = next(mtg_metamers_iterator)
                        if int(self._shared_mtg.index(mtg_metamer_vid)) == turgorgrowth_phytomer_index:
                            break
                    if turgorgrowth_phytomer.hiddenzone is not None:
                        mtg_hiddenzone_label = turgorgrowth_converter.TURGORGROWTH_CLASSES_TO_DATAFRAME_ORGANS_MAPPING[turgorgrowth_model.HiddenZone]
                        if mtg_hiddenzone_label not in self._shared_mtg.get_vertex_property(mtg_metamer_vid):
                            # Add a property describing the hiddenzone to the current metamer of the MTG
                            self._shared_mtg.property(mtg_hiddenzone_label)[mtg_metamer_vid] = {}
                        # Update the property describing the hiddenzone of the current metamer in the MTG
                        mtg_hiddenzone_properties = self._shared_mtg.get_vertex_property(mtg_metamer_vid)[mtg_hiddenzone_label]
                        mtg_hiddenzone_properties.update(turgorgrowth_phytomer.hiddenzone.__dict__)
                    for mtg_organ_vid in self._shared_mtg.components_iter(mtg_metamer_vid):
                        mtg_organ_label = self._shared_mtg.label(mtg_organ_vid)
                        if mtg_organ_label not in MTG_TO_TURGORGROWTH_PHYTOMERS_ORGANS_MAPPING: continue
                        turgorgrowth_organ = getattr(turgorgrowth_phytomer, TURGORGROWTH_ATTRIBUTES_MAPPING[MTG_TO_TURGORGROWTH_PHYTOMERS_ORGANS_MAPPING[mtg_organ_label]])
                        if turgorgrowth_organ is None:
                            continue
                        # element scale
                        for mtg_element_vid in self._shared_mtg.components_iter(mtg_organ_vid):
                            mtg_element_label = self._shared_mtg.label(mtg_element_vid)
                            if mtg_element_label not in turgorgrowth_converter.DATAFRAME_TO_TURGORGROWTH_ELEMENTS_NAMES_MAPPING: continue
                            turgorgrowth_element = getattr(turgorgrowth_organ, turgorgrowth_converter.DATAFRAME_TO_TURGORGROWTH_ELEMENTS_NAMES_MAPPING[mtg_element_label])
                            turgorgrowth_element_property_names = [property_name for property_name in turgorgrowth_simulation.Simulation.ELEMENTS_RUN_VARIABLES if hasattr(turgorgrowth_element, property_name)]
                            for turgorgrowth_element_property_name in turgorgrowth_element_property_names:
                                turgorgrowth_element_property_value = getattr(turgorgrowth_element, turgorgrowth_element_property_name)
                                self._shared_mtg.property(turgorgrowth_element_property_name)[mtg_element_vid] = turgorgrowth_element_property_value

                        # update of organ scale from elements
                        new_mtg_element_labels = {}
                        for new_element_vid in self._shared_mtg.components_iter(mtg_organ_vid):
                            new_element_label = self._shared_mtg.label(new_element_vid)
                            new_mtg_element_labels[new_element_label] = new_element_vid

                        if mtg_organ_label == 'blade' and 'LeafElement1' in new_mtg_element_labels.keys():
                            organ_visible_length = self._shared_mtg.property('length')[new_mtg_element_labels['LeafElement1']]
                            self._shared_mtg.property('visible_length')[mtg_organ_vid] = organ_visible_length
                        elif mtg_organ_label == 'sheath' and 'StemElement' in new_mtg_element_labels.keys():
                            organ_visible_length = self._shared_mtg.property('length')[new_mtg_element_labels['StemElement']]
                            ## TO DO : BOXE OR CYLINDER ???
                            # self._shared_mtg.property('diameter')[mtg_organ_vid] = self._shared_mtg.property('width')[new_mtg_element_labels['StemElement']] / pi
                            self._shared_mtg.property('visible_length')[mtg_organ_vid] = organ_visible_length
                        elif mtg_organ_label == 'internode' and 'StemElement' in new_mtg_element_labels.keys():
                            organ_visible_length = self._shared_mtg.property('length')[new_mtg_element_labels['StemElement']]
                            self._shared_mtg.property('visible_length')[mtg_organ_vid] = organ_visible_length
                        else:
                            organ_visible_length = 0

                        if 'HiddenElement' in new_mtg_element_labels.keys():
                            organ_hidden_length = self._shared_mtg.property('length')[new_mtg_element_labels['HiddenElement']]
                        else:
                            organ_hidden_length = 0

                        total_organ_length = organ_visible_length + organ_hidden_length
                        self._shared_mtg.property('length')[mtg_organ_vid] = total_organ_length

    def _update_shared_dataframes(self, turgorgrowth_axes_data_df=None, turgorgrowth_hiddenzones_data_df=None, turgorgrowth_elements_data_df=None, turgorgrowth_organs_data_df=None, turgorgrowth_soil_data_df=None):
        """
        Update the dataframes shared between all models from the inputs dataframes or the outputs dataframes of the turgorgrowth model.
        """

        for turgorgrowth_data_df, \
            shared_inputs_outputs_indexes, \
            shared_inputs_outputs_df in ((turgorgrowth_axes_data_df, turgorgrowth_simulation.Simulation.AXES_INDEXES, self._shared_axes_inputs_outputs_df),
                                         (turgorgrowth_hiddenzones_data_df, turgorgrowth_simulation.Simulation.HIDDENZONE_INDEXES, self._shared_hiddenzones_inputs_outputs_df),
                                         (turgorgrowth_elements_data_df, turgorgrowth_simulation.Simulation.ELEMENTS_INDEXES, self._shared_elements_inputs_outputs_df),
                                         (turgorgrowth_organs_data_df, turgorgrowth_simulation.Simulation.ORGANS_INDEXES, self._shared_organs_inputs_outputs_df),
                                         (turgorgrowth_soil_data_df, turgorgrowth_simulation.Simulation.SOIL_INDEXES, self._shared_soil_inputs_outputs_df)):
            # if turgorgrowth_data_df is None: continue
            tools.combine_dataframes_inplace(turgorgrowth_data_df, shared_inputs_outputs_indexes, shared_inputs_outputs_df)
