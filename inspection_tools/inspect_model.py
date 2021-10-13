import glob
import pickle
from argparse import ArgumentParser
import logging
import numpy as np
import time

from obstacle_detector.niche_image_creator import NicheImageCreator
from poet_distributed.es import ESOptimizer
from poet_distributed.niches.box2d.cppn import CppnEnvParams
from poet_distributed.niches.box2d.model import Model, simulate
from poet_distributed.niches.box2d.env import bipedhard_custom, Env_config
from inspection_tools.file_utilities import get_cppn_file_list, get_cppn_file_iterator, \
    get_model_file_list, get_model_file_iterator, get_latest_cppn_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_model(model_file, cppn_genome_path):
    """
    Load and inspect the performance of an agent in a given environment
    :param model_file: The full path of the .json file containing the agent model to be inspected.
    :param cppn_genome_path: The full path of the pickle file containing the CPPN that the agent will be tested in
    :return pos: (x, y) Tuple containing the position of an agents death, which can be input to (draw_
    """

    # set master_seed
    seed = -1  # if seed should not be used
    # seed = 42  # if seed should be used

    default_environment = Env_config(
        name='flat',
        ground_roughness=0,
        pit_gap=[],
        stump_width=[],
        stump_height=[],
        stump_float=[],
        stair_height=[],
        stair_width=[],
        stair_steps=[])

    cppn_params = CppnEnvParams(genome_path=cppn_genome_path)

    test_model = Model(bipedhard_custom)
    test_model.make_env(seed=seed, env_config=default_environment)
    test_model.load_model(filename=model_file)
    _, _, pos = simulate(test_model, seed=seed, max_len=2000,
                         train_mode=True, render_mode=True, num_episode=1,
                         env_config_this_sim=default_environment, env_params=cppn_params,
                         get_pos_at_death=True)

    return pos


def draw_terrain_on_position(cppn_genome_path, pos):
    """
    Draws the CPPN encoding of the position given, typically a location of death
    :param cppn_genome_path: Full path of CPPN genome
    :param pos: tuple containing the x,y coordinates of a location in the environment
    :return None:
    """
    img_creator = NicheImageCreator(cppn_genome_path)
    img_creator.current_image = img_creator.create_image(mid_x=pos[0],
                                                         in_width=8, in_height=8,
                                                         out_width=64, out_height=64)
    img_creator.show_image()


def main(model_file, cppn_genome_path):

    point_of_termination = inspect_model(model_file, cppn_genome_path)
    # draw_terrain_on_position(cppn_genome_path, point_of_termination)


if __name__ == "__main__":
    """
    Run a custom test of an agent and a cppn, or multiple sequentially. 
    """
    test_run_name = 'okt8_overnight'

    for current_agent_model_json in get_model_file_iterator(training_run=test_run_name):
        current_optimizer_log_file = current_agent_model_json.split('.best.json')[0] + '.log'

        current_cppn_genome_file = get_latest_cppn_file(training_run=test_run_name,
                                                        optimizer_log_file=current_optimizer_log_file)

        print("\n\nNow running agent: {} \non environment: {}".format(current_agent_model_json,
                                                                      current_cppn_genome_file))

        main(model_file=current_agent_model_json, cppn_genome_path=current_cppn_genome_file)
