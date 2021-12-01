"""
Master script for collecting dataset from Enhanced POET
"""

from argparse import ArgumentParser
import logging
import numpy as np
import mlflow as mlf
from poet_distributed.es import initialize_master_fiber
from poet_distributed.poet_algo import MultiESOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_main(args):

    initialize_master_fiber()

    # set master_seed
    np.random.seed(args.master_seed)

    optimizer_zoo = MultiESOptimizer(args=args)

    optimizer_zoo.optimize(iterations=args.n_iterations,
                           propose_with_adam=args.propose_with_adam,
                           reset_optimizer=True,
                           checkpointing=args.checkpointing,
                           steps_before_transfer=args.steps_before_transfer)


def main():
    parser = ArgumentParser()
    parser.add_argument('log_file')
    parser.add_argument('niche_file')
    parser.add_argument('dataset_folder')
    parser.add_argument('--save_to_dataset', default=True)
    parser.add_argument('--distance_threshold', type=float, default=3)
    parser.add_argument('--init', default='random')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.9999)
    parser.add_argument('--lr_limit', type=float, default=0.001)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--noise_decay', type=float, default=0.999)
    parser.add_argument('--noise_limit', type=float, default=0.01)
    parser.add_argument('--l2_coeff', type=float, default=0.01)
    parser.add_argument('--batches_per_chunk', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--eval_batches_per_step', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--n_iterations', type=int, default=200)
    parser.add_argument('--steps_before_transfer', type=int, default=25)
    parser.add_argument('--max_children', type=int, default=8)
    parser.add_argument('--max_admitted', type=int, default=1)
    parser.add_argument('--master_seed', type=int, default=111)
    parser.add_argument('--mc_lower', type=int, default=25)
    parser.add_argument('--mc_upper', type=int, default=340)
    parser.add_argument('--repro_threshold', type=int, default=200)
    parser.add_argument('--max_num_envs', type=int, default=100)
    parser.add_argument('--normalize_grads_by_noise_std', action='store_true', default=False)
    parser.add_argument('--propose_with_adam', action='store_true', default=False)
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--adjust_interval', type=int, default=4)
    parser.add_argument('--returns_normalization', default='normal')
    parser.add_argument('--stochastic', action='store_true', default=False)
    parser.add_argument('--envs', nargs='+')
    parser.add_argument('--start_from', default=None)  # Json file to start from

    args = parser.parse_args()
    logger.info(args)

    for arg_name in args.__dict__:
        mlf.log_param(arg_name, args.__dict__[arg_name])

    run_main(args)


if __name__ == "__main__":
    main()
