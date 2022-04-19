import logging
import numpy as np
import random

from obstacle_detector.niche_image_creator import NicheImageCreator
from poet_distributed.niches.box2d.cppn import CppnEnvParams
from poet_distributed.niches.box2d.model import Model, simulate
from poet_distributed.niches.box2d.env import bipedhard_custom, Env_config
from inspection_tools.file_utilities import get_model_file_iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_novel_cppn_environments(n_environments=10, n_mutations=10, seed=-1):
    """
    Generates a set of novel environments from cppn-encodings.
    :param n_environments: number of environments to generate
    :param n_mutations: number of mutations on each environment encoding. High values will give more varied terrain.
    :param seed: When equal or above 0, sets random seed for environment generation.
    :return: List of novel cppn parameters
    """

    if seed >= 0:
        logger.debug('Setting seed to {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)

    environments = []
    for i in range(n_environments):
        print("Mutating env {} - {} times".format(i, n_mutations))
        cppn_params = CppnEnvParams()
        for j in range(n_mutations):
            cppn_params = cppn_params.get_mutated_params("/uio/hume/student-u31/eirikolb/tmp", save_cppn=False)
        environments.append(cppn_params)
        print("Done with env {}".format(i))
    return environments


def run_simulation(model_file, cppn_genome, seed=-1, max_len=2000):
    """
    Load and inspect the performance of an agent in a given environment
    :param model_file: The full path of the .json file containing the agent model to be inspected.
    :param cppn_genome: Either the full path of the pickle file containing the CPPN that the agent will be tested in,
    or the CppnEnvParams to be used directly
    :param seed: int, when bigger than or equal to zero, sets the random seed
    :param max_len: int, number of steps to simulate the agent before termination
    :return pos: (x, y) Tuple containing the position of an agents death, which can be input to (draw_
    """

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

    if isinstance(cppn_genome, CppnEnvParams):
        cppn_params = cppn_genome
    elif isinstance(cppn_genome, str):
        cppn_params = CppnEnvParams(genome_path=cppn_genome)
    else:
        raise IOError()

    test_model = Model(bipedhard_custom)
    test_model.make_env(seed=seed, env_config=default_environment)
    test_model.load_model(filename=model_file)
    _, _, info = simulate(test_model, seed=seed, train_mode=True, render_mode=True, num_episode=1, max_len=max_len,
                          env_config_this_sim=default_environment, env_params=cppn_params)

    return info


def draw_terrain_on_position(cppn_genome_path, pos):
    """
    Draws the CPPN encoding of the position given, typically a location of death
    :param cppn_genome_path: Full path of CPPN genome
    :param pos: tuple containing the x,y coordinates of a location in the environment
    :return None:
    """
    img_creator = NicheImageCreator(cppn_genome_path)
    img_creator.current_image = img_creator.create_image(mid_x=pos.x,
                                                         in_width=8, in_height=8,
                                                         out_width=64, out_height=64)
    img_creator.show_image()


def main(model_file, cppn_genome_path, seed=-1):
    """
    Run simulation on given model and environment, and draw the location of stumble if the agent dies.
    """
    info = run_simulation(model_file, cppn_genome_path, seed=seed)
    if info['game_over']:
        location_of_death = info['pos']
        draw_terrain_on_position(cppn_genome_path, location_of_death)


if __name__ == "__main__":
    """
    Generate a set of novel cppn encodings, then run some list of agents on these.
    If an agent dies, the location of its stumble is drawn
    """
    test_run_name = 'sep30_overnight'
    test_seeds = [12]

    for test_seed in test_seeds:
        novel_environments = generate_novel_cppn_environments(n_environments=2, n_mutations=50, seed=test_seed)
        print("done")

        for current_agent_model_json in get_model_file_iterator(training_run=test_run_name):
            for current_cppn_genome in novel_environments:
                main(model_file=current_agent_model_json, cppn_genome_path=current_cppn_genome, seed=test_seed)
