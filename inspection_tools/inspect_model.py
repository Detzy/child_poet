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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_cppn_attributes(cppn_genome_path):
    cppn_params = CppnEnvParams(genome_path=cppn_genome_path)
    # print(cppn_params.cppn_genome)
    print(cppn_params.cppn_genome)
    print(cppn_params.cppn_genome.key)
    print()


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
    # parser = ArgumentParser()
    # parser.add_argument('--log_file', default='~/tmp/logs/$experiment')
    # parser.add_argument('--init', default='random')
    # parser.add_argument('--learning_rate', type=float, default=0.01)
    # parser.add_argument('--lr_decay', type=float, default=0.9999)
    # parser.add_argument('--lr_limit', type=float, default=0.001)
    # parser.add_argument('--noise_std', type=float, default=0.1)
    # parser.add_argument('--noise_decay', type=float, default=0.999)
    # parser.add_argument('--noise_limit', type=float, default=0.01)
    # parser.add_argument('--l2_coeff', type=float, default=0.01)
    # parser.add_argument('--batches_per_chunk', type=int, default=50)
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--eval_batch_size', type=int, default=1)
    # parser.add_argument('--eval_batches_per_step', type=int, default=50)
    # parser.add_argument('--num_workers', type=int, default=20)
    # parser.add_argument('--n_iterations', type=int, default=200)
    # parser.add_argument('--steps_before_transfer', type=int, default=25)
    # parser.add_argument('--master_seed', type=int, default=111)
    # parser.add_argument('--mc_lower', type=int, default=25)
    # parser.add_argument('--mc_upper', type=int, default=340)
    # parser.add_argument('--repro_threshold', type=int, default=200)
    # parser.add_argument('--max_num_envs', type=int, default=100)
    # parser.add_argument('--normalize_grads_by_noise_std', action='store_true', default=False)
    # parser.add_argument('--propose_with_adam', action='store_true', default=False)
    # parser.add_argument('--checkpointing', action='store_true', default=False)
    # parser.add_argument('--adjust_interval', type=int, default=4)
    # parser.add_argument('--returns_normalization', default='normal')
    # parser.add_argument('--stochastic', action='store_true', default=False)
    # parser.add_argument('--envs', nargs='+')
    # parser.add_argument('--start_from', default=model_file)  # Json file to start from

    # args = parser.parse_args()
    # logger.info(args)

    point_of_termination = inspect_model(model_file, cppn_genome_path)
    # draw_terrain_on_position(cppn_genome_path, point_of_termination)


if __name__ == "__main__":
    """
    Run a custom test of an agent and a cppn, or multiple sequentially. 
    """
    test_run_name = 'okt4_overnight'
    # agent_name = 'flat'
    agent_model_folder = r'/uio/hume/student-u31/eirikolb/tmp/logs/poet_{}/'.format(test_run_name)
    # agent_model_json_name = r'poet_{}.{}.best.json'.format(test_run_name, agent_name)
    agent_model_json_name = r'poet_*.best.json'
    agent_model_json = agent_model_folder + agent_model_json_name

    time_string = True  # String, or False
    cppn_genome_folder = '/uio/hume/student-u31/eirikolb/tmp/niche_encodings/poet_{}/'.format(test_run_name)
    # cppn_genome_file_name = 'genome_{}.pickle'.format(time_string)  # test just one
    cppn_genome_file_name = 'genome_*.pickle'  # loop all
    cppn_genome_file = cppn_genome_folder + cppn_genome_file_name

    for current_agent_model_json in glob.iglob(agent_model_json):
        for current_cppn_genome_file in sorted(glob.glob(cppn_genome_file), reverse=True):
            print("Now running agent: {} \non environment: {}".format(current_agent_model_json, current_cppn_genome_file))
            if time_string:
                main(model_file=current_agent_model_json, cppn_genome_path=current_cppn_genome_file)
            else:
                main(model_file=current_agent_model_json, cppn_genome_path=False)
            # print_cppn_attributes(current_cppn_genome_file)
            # input()
