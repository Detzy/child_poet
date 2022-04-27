# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import numpy as np
import mlflow as mlf
import json
from obstacle_detector.niche_image_creator import NicheImageCreator
from poet_distributed.es import ESOptimizer
from poet_distributed.es import initialize_worker_fiber
from collections import OrderedDict
from poet_distributed.niches.box2d.env import Env_config
from poet_distributed.niches.box2d.cppn import CppnEnvParams
from poet_distributed.reproduce_ops import Reproducer
from poet_distributed.novelty import compute_novelty_vs_archive

logger = logging.getLogger(__name__)


CPPN_MUTATION_RATE = 3


def load_agent_model(filename):
    with open(filename) as f:
        data = json.load(f)
    print('loading file %s' % filename)
    model_params = np.array(data[0])
    return model_params


def construct_niche_fns_from_env(args, env, env_params, seed, img_creator=None):
    def niche_wrapper(configs, env_params, seed):  # force python to make a new lexical scope
        def make_niche():
            from poet_distributed.niches import Box2DNiche
            return Box2DNiche(env_configs=configs,
                              env_params=env_params,
                              seed=seed,
                              init=args.init,
                              stochastic=args.stochastic,
                              img_creator=img_creator)

        return make_niche

    niche_name = env.name
    configs = (env,)

    return niche_name, niche_wrapper(list(configs), env_params, seed)


class MultiESOptimizer:
    """
    The primary component of Enhanced POET. This class is responsible for running the primary loop in Enhanced,
    continually creating and optimizing environments and their paired agents.
    """

    def __init__(self, args):

        self.args = args
        self.predict_simulation = self.args.run_child_poet
        self.agent_trackers = None
        if self.predict_simulation:
            self.agent_trackers = {}

        import fiber as mp

        mp_ctx = mp.get_context('spawn')
        manager = mp_ctx.Manager()
        self.manager = manager
        self.fiber_shared = {
            "niches": manager.dict(),
            "thetas": manager.dict(),
        }
        self.fiber_pool = mp_ctx.Pool(args.num_workers, initializer=initialize_worker_fiber,
                                      initargs=(self.fiber_shared["thetas"],
                                                self.fiber_shared["niches"],
                                                ))

        self.ANNECS = 0
        self.env_registry = OrderedDict()
        self.env_archive = OrderedDict()
        self.env_reproducer = Reproducer(args)
        self.optimizers = OrderedDict()
        self.archived_optimizers = OrderedDict()

        env = Env_config(
            name='flat',
            ground_roughness=0,
            pit_gap=[],
            stump_width=[],
            stump_height=[],
            stump_float=[],
            stair_height=[],
            stair_width=[],
            stair_steps=[])

        agent_model = None
        if args.start_from is not None:
            agent_model = load_agent_model(args.start_from)

        params = CppnEnvParams()
        self.add_optimizer(env=env, cppn_params=params, seed=args.master_seed, model_params=agent_model)

    def create_optimizer(self, env, cppn_params, seed, created_at=0, model_params=None, is_candidate=False):
        """
        Create a new optimizer connected to a given environment description.
        :param env: The parameters associated with the new environment
        :param cppn_params: The cppn_parameters used to encode the new environment
        :param seed: Seed used to generate the environment
        :param created_at: The iteration when this optimizer created
        :param model_params: The parameters for the agent model that will be paired with the environment
        :param is_candidate: I don't know exactly what this is, but when this is false, some info about the
        optimizer is logged.
        :return: ESOptimiser object
        """
        assert env is not None
        assert cppn_params is not None

        img_creator = None
        if self.args.save_to_dataset:
            img_creator = NicheImageCreator(cppn_params=None, dataset_folder=self.args.dataset_folder,
                                            distance_threshold=self.args.distance_threshold)
        elif self.args.run_child_poet:
            img_creator = NicheImageCreator(cppn_params=None, dataset_folder=None,
                                            distance_threshold=None)

        optim_id, niche_fn = construct_niche_fns_from_env(args=self.args, env=env, env_params=cppn_params,
                                                          seed=seed, img_creator=img_creator)

        niche = niche_fn()
        if model_params is not None:
            theta = np.array(model_params)
        else:
            theta = niche.initial_theta()
        assert optim_id not in self.optimizers.keys()

        return ESOptimizer(
            optim_id=optim_id,
            fiber_pool=self.fiber_pool,
            fiber_shared=self.fiber_shared,
            theta=theta,
            make_niche=niche_fn,
            learning_rate=self.args.learning_rate,
            lr_decay=self.args.lr_decay,
            lr_limit=self.args.lr_limit,
            batches_per_chunk=self.args.batches_per_chunk,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            eval_batches_per_step=self.args.eval_batches_per_step,
            l2_coeff=self.args.l2_coeff,
            noise_std=self.args.noise_std,
            noise_decay=self.args.noise_decay,
            normalize_grads_by_noise_std=self.args.normalize_grads_by_noise_std,
            returns_normalization=self.args.returns_normalization,
            noise_limit=self.args.noise_limit,
            log_file=self.args.log_file,
            created_at=created_at,
            is_candidate=is_candidate,
            predict_simulation=self.args.run_child_poet,
            omit_simulation=self.args.omit_simulation,
            agent_tracker_success_reward=self.args.child_success_reward,
            agent_tracker_certainty_threshold=self.args.agent_tracker_certainty_threshold,
            agent_trackers=self.agent_trackers,
        )

    def add_optimizer(self, env, cppn_params, seed, created_at=0, model_params=None):
        """
        Create a new optimizer connected to a given environment description,
        then add it to the collections of current and archived optimizers.

        :param env: The parameters associated with the new environment
        :param cppn_params: The cppn_parameters used to encode the new environment
        :param seed: Seed used to generate the environment
        :param created_at: the iteration when this niche is created
        :param model_params: I don't know exactly what this is, but it is the theta which is given elsewhere,
        and is related to changes between environments, I think.
        :return: None
        """

        o = self.create_optimizer(env, cppn_params, seed, created_at, model_params)
        optim_id = o.optim_id
        self.optimizers[optim_id] = o

        assert optim_id not in self.env_registry.keys()
        assert optim_id not in self.env_archive.keys()

        self.env_registry[optim_id] = (env, cppn_params)
        self.env_archive[optim_id] = (env, cppn_params)

        cppn_params.save_genome(self.args.niche_file)

    def archive_optimizer(self, optim_id):
        """
        Move optimiser from current optimizers into the archived ones

        :param optim_id: id of optimizer to archive
        :return: None
        """
        assert optim_id in self.optimizers.keys()
        # assume optim_id == env_id for single_env niches
        o = self.optimizers.pop(optim_id)
        assert optim_id in self.env_registry.keys()
        self.env_registry.pop(optim_id)
        logger.info('Archived {} '.format(optim_id))
        self.archived_optimizers[optim_id] = o

    def ind_es_step(self, iteration):
        """
        I think this is called "induce evolutionary strategy step"?
        For each optimizer, perform one step of ES, and update the optimizers agent model accordingly

        (I think this previously did not use parallelization properly, so I changed it from the original.
        This change has later been implemented similarly other places in the POET library. I hope I did it right)

        We decide that during this process, data should not be collected. This helps remove failures from
        incapable agents.

        :param iteration: int, current iteration
        :return: None
        """
        tasks = [o.start_step(gather_obstacle_dataset=False) for o in self.optimizers.values()]
        self_eval_tasks = []
        for optimizer, task in zip(self.optimizers.values(), tasks):

            # Update the theta of each optimizer based on training results
            optimizer.theta, stats = optimizer.get_step(task)
            self_eval_tasks.append((
                optimizer.start_theta_eval(optimizer.theta, agent_tracker=None,
                                           gather_obstacle_dataset=False, predict_simulation=False),
                optimizer))

        for self_eval_task, optimizer in self_eval_tasks:
            self_eval_stats = optimizer.get_theta_eval(self_eval_task)

            logger.info('Iter={} Optimizer {} theta_mean {} best po {} iteration spent {}'.format(
                iteration, optimizer.optim_id, self_eval_stats.eval_returns_mean,
                stats.po_returns_max, iteration - optimizer.created_at))

            optimizer.update_dicts_after_es(stats=stats,
                                            self_eval_stats=self_eval_stats)

    def transfer(self, propose_with_adam, checkpointing, reset_optimizer):
        """
        This method is responsible for the transfer of agent models across environments.
        For every agent, test it in all other environments,
        and if they exceed the currently paired agent's score, they are moved to a list of candiate transfers.
        If the candidates also exceed the paired agent of an environment after training in the environment,
        the best candidate replaces the currently paired agent.
        (I think this previously did not use parallelization properly, so I changed it from the original.
        This change has later been implemented similarly other places in the POET library. I hope I did it right.
        (I did not do it right. It was fine all along.))
        """
        logger.info('Computing direct transfers...')
        proposal_targets = {}
        for source_optim in self.optimizers.values():
            source_tasks = []
            proposal_targets[source_optim] = []
            for target_optim in [o for o in self.optimizers.values() if o is not source_optim]:
                task = target_optim.start_theta_eval(
                    source_optim.theta,
                    source_optim.agent_tracker,
                    gather_obstacle_dataset=self.args.save_to_dataset,
                    predict_simulation=self.predict_simulation)
                source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                stats = target_optim.get_theta_eval(task)

                try_proposal = target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                                                                        source_optim_theta=source_optim.theta,
                                                                        stats=stats, keyword='theta')
                if try_proposal:
                    proposal_targets[source_optim].append(target_optim)

        logger.info('Computing proposal transfers...')
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                 if o is not source_optim]:
                if target_optim in proposal_targets[source_optim]:
                    task = target_optim.start_step(source_optim.theta,
                                                   gather_obstacle_dataset=self.args.save_to_dataset)
                    source_tasks.append((task, target_optim))

            proposed_tasks = []
            for task, target_optim in source_tasks:
                proposed_theta, _ = target_optim.get_step(
                    task, propose_with_adam=propose_with_adam, propose_only=True)

                proposed_tasks.append((
                    target_optim.start_theta_eval(
                        proposed_theta,
                        agent_tracker=None,
                        gather_obstacle_dataset=self.args.save_to_dataset,
                        predict_simulation=False
                    ), target_optim))

            for proposed_task, target_optim in proposed_tasks:
                proposal_eval_stats = target_optim.get_theta_eval(proposed_task)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                                                         source_optim_theta=proposed_theta,
                                                         stats=proposal_eval_stats, keyword='proposal')

        logger.info('Considering transfers...')
        for o in self.optimizers.values():
            o.pick_proposal(checkpointing, reset_optimizer)

    def check_optimizer_status(self, iteration):
        """
        Check all optimizers for whether or not the score of their agent in their paired niche is above a threshold,
        and if they are, add them to candidates for reproduction. Returns this list of reproduction candidates,
        as well as an empty list of deletion candidates.

        :param iteration: Unused parameter
        :return: Two lists, candidates for reproduction and candidates for deletions
        """
        logger.info("health_check")
        repro_candidates, delete_candidates = [], []
        for optim_id in self.env_registry.keys():
            o = self.optimizers[optim_id]
            logger.info("niche {} created at {} start_score {} current_self_evals {}".format(
                optim_id, o.created_at, o.start_score, o.self_evals))
            if o.self_evals >= self.args.repro_threshold:
                repro_candidates.append(optim_id)

        # logger.debug("candidates to reproduce")
        # logger.debug(repro_candidates)
        # logger.debug("candidates to delete")
        # logger.debug(delete_candidates)
        logger.info("candidates to reproduce")
        logger.info(repro_candidates)
        logger.info("candidates to delete")
        logger.info(delete_candidates)

        return repro_candidates, delete_candidates

    def pass_dedup(self, env_config):
        """
        Check if the environment name is available, or already taken.

        :return: True if env name is available, else False
        """
        if env_config.name in self.env_registry.keys():
            logger.debug("active env already. reject!")
            return False
        else:
            return True

    def pass_mc(self, score):
        """
        Check if the score of an agent in an environment passes the minimal criterion

        :return: Bool, whether or not the score passes the minimal criterion
        """
        if score < self.args.mc_lower or score > self.args.mc_upper:
            return False
        else:
            return True

    def get_new_env(self, list_repro):
        """
        Pick a random environment from the given list, and mutate its parameters.

        :param list_repro: List of candidates for reproduction
        :return: Tuple (env_config, cppn_params, seed, parent_optimiser_id)
        """
        optim_id = self.env_reproducer.pick(list_repro)
        assert optim_id in self.optimizers.keys()
        assert optim_id in self.env_registry.keys()
        parent_env_config, parent_cppn_params = self.env_registry[optim_id]
        child_env_config = self.env_reproducer.mutate(parent_env_config, no_mutate=True)  # Only sets a new name

        # Generate a new environment by mutating the parent environment 'CPPN_MUTATION_RATE' times
        child_cppn_params = parent_cppn_params
        for i in range(CPPN_MUTATION_RATE):
            child_cppn_params = child_cppn_params.get_mutated_params(cppn_path_string=self.args.niche_file)

        logger.info("we pick to mutate: {} and we got {} back".format(optim_id, child_env_config.name))
        logger.debug("parent")
        logger.debug(parent_env_config)
        logger.debug("child")
        logger.debug(child_env_config)

        seed = np.random.randint(1000000)
        return child_env_config, child_cppn_params, seed, optim_id

    def get_child_list(self, parent_list, max_children):
        """
        Create a list containing up to max_children new child environments,
        sorted descending by their PATA-EC novelty.
        Child environments will only be added if the agent of the parent environment passes the minimal criterion func

        :param parent_list: list of potential parents,
        which will be randomly selected from when creating new environments
        :param max_children: int, max number of children reproduced
        :return: list of child environments that passed minimal criterion, sorted descending by their PATA-EC novelty
        """
        child_list = []

        mutation_trial = 0
        while mutation_trial < max_children:
            # Get new potential child environment
            new_env_config, new_cppn_params, seed, parent_optim_id = self.get_new_env(parent_list)
            mutation_trial += 1
            if self.pass_dedup(new_env_config):
                # Create a new optimizer with the child candidate. If the agent paired with the parent env
                # passes the minimal criterion function (is neither too good nor too bad), it is added to child_list
                o = self.create_optimizer(new_env_config, new_cppn_params, seed, is_candidate=True)
                eval_stats = o.evaluate_theta(self.optimizers[parent_optim_id].theta,
                                              self.optimizers[parent_optim_id].agent_tracker,
                                              gather_obstacle_dataset=self.args.save_to_dataset)
                score = eval_stats.eval_returns_mean

                if self.pass_mc(score):
                    novelty_score = compute_novelty_vs_archive(self.archived_optimizers, self.optimizers, o, k=5,
                                                               low=self.args.mc_lower, high=self.args.mc_upper,
                                                               gather_obstacle_dataset=self.args.save_to_dataset)
                    logger.debug("{} passed mc, novelty score {}".format(score, novelty_score))
                    child_list.append((new_env_config, new_cppn_params, seed, parent_optim_id, novelty_score))
                del o

        # sort child list according to novelty for high to low
        child_list = sorted(child_list, key=lambda x: x[4], reverse=True)
        return child_list

    def adjust_envs_niches(self, iteration, steps_before_adjust, max_num_envs=None, max_children=8, max_admitted=1):
        """
        Method for evolving and adding environment niches.
        If iteration is a multiple of steps_before_adjust, perform:
        - Check what niches are ready for reproduction
        - Update pata-ec score in all niches
        - Produce a sorted list of child niches, evolved from a random selection of the current niches.
        - For each child niche, cross-evaluate all current and archived agent models in it, and find the best.
        --- If the best current agent model passes the minimal criterion, pair the new niche with that agent
        --- If the best archived agent model also passes the minimal criterion, the ANNECS measure is increased
        - If after this process more optimizers exist than is allowed, remove the oldest until no longer above the limit

        :param iteration: The current iteration of optimization.
        :param steps_before_adjust: Run this method only on iterations that are a multiple of this number.
        :param max_num_envs: Maximum number of environment-agent pairs that can exist at a time.
        :param max_children: Maximum number of child niches that can be evolved
        :param max_admitted: Maximum number of the new child niches that can be added to the population this iteration
        :return: None
        """

        if iteration > 0 and iteration % steps_before_adjust == 0:
            # - Check what niches are ready for reproduction
            list_repro, list_delete = self.check_optimizer_status(iteration)

            if len(list_repro) == 0:
                return

            logger.info("list of niches to reproduce")
            logger.info(list_repro)
            logger.info("list of niches to delete")
            logger.info(list_delete)

            # - Update pata-ec score in all niches
            for optim in self.optimizers.values():
                optim.update_pata_ec(self.archived_optimizers, self.optimizers, self.args.mc_lower, self.args.mc_upper,
                                     gather_obstacle_dataset=self.args.save_to_dataset)

            for optim in self.archived_optimizers.values():
                optim.update_pata_ec(self.archived_optimizers, self.optimizers, self.args.mc_lower, self.args.mc_upper,
                                     gather_obstacle_dataset=self.args.save_to_dataset)

            # - Produce a sorted list of child niches, evolved from a random selection of the current niches.
            child_list = self.get_child_list(list_repro, max_children)

            if child_list is None or len(child_list) == 0:
                logger.info("mutation to reproduce env FAILED!!!")
                return

            # - For each child niche, cross-evaluate current and archived agent models in it, and find the best agent.
            admitted = 0
            for child in child_list:
                new_env_config, new_cppn_params, seed, _, _ = child
                # targeted transfer
                o = self.create_optimizer(new_env_config, new_cppn_params, seed, is_candidate=True)
                score_child, theta_child = \
                    o.evaluate_transfer(self.optimizers, gather_obstacle_dataset=self.args.save_to_dataset)
                score_archive, _ = o.evaluate_transfer(self.archived_optimizers, evaluate_proposal=False,
                                                       gather_obstacle_dataset=self.args.save_to_dataset)
                del o

                # -- If the best current agent model passes the minimal criterion, pair the new niche with that agent
                if self.pass_mc(score_child):  # check mc
                    self.add_optimizer(env=new_env_config, cppn_params=new_cppn_params, seed=seed, created_at=iteration,
                                       model_params=np.array(theta_child))
                    admitted += 1

                    # --- If the best archived agent model also passes the minimal criterion,
                    # the ANNECS measure is increased
                    if score_archive is not None and self.pass_mc(score_archive):
                        self.ANNECS += 1
                        mlf.log_metric("ANNECS", self.ANNECS)
                        mlf.log_metric("iteration", self.iteration)
                    if admitted >= max_admitted:
                        break

            # - If after this process more optimizers exist than is allowed,
            # remove the oldest until no longer above the limit
            if max_num_envs is not None and len(self.optimizers) > max_num_envs:
                num_removals = len(self.optimizers) - max_num_envs
                self.remove_oldest(num_removals)

    def remove_oldest(self, num_removals):
        """
        Method for finding and removing the oldest environments (niches).

        :param num_removals: Number of niches to remove
        """
        list_delete = []
        for optim_id in self.env_registry.keys():
            if len(list_delete) < num_removals:
                list_delete.append(optim_id)
            else:
                break

        for optim_id in list_delete:
            self.archive_optimizer(optim_id)

    def optimize(self, iterations=200,
                 steps_before_transfer=25,
                 propose_with_adam=False,
                 checkpointing=False,
                 reset_optimizer=True):
        """
        Master loop for the optimisation phase.
        Each loop, we train and adjust agents.
        In some iterations, with intervals defined by the parameters args.adjust_interval and steps_before_t..., we do:
        - Evolve new niches
        - Test performance of agents in all other niches, and transfer those that outperform
        - Save to logger what iteration transfers where made
        - Note: evolving niches happens as often as, or rarer than transfers, based on args.adjust_interval
        """
        for iteration in range(iterations):
            # evolve new niches
            self.adjust_envs_niches(iteration, self.args.adjust_interval * steps_before_transfer,
                                    max_num_envs=self.args.max_num_envs, max_children=self.args.max_children,
                                    max_admitted=self.args.max_admitted)

            for o in self.optimizers.values():
                o.clean_dicts_before_iter()

            # perform evolution to train and adjust agents
            self.ind_es_step(iteration=iteration)

            # perform transfer
            if len(self.optimizers) > 1 and iteration % steps_before_transfer == 0:
                self.transfer(propose_with_adam=propose_with_adam,
                              checkpointing=checkpointing,
                              reset_optimizer=reset_optimizer)

            # log what iterations transfers were made
            if iteration % steps_before_transfer == 0:
                for o in self.optimizers.values():
                    o.save_to_logger(iteration)
