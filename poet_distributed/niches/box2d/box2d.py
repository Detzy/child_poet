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

# from obstacle_detector.niche_image_creator import NicheImageCreator
import logging
from ..core import Niche
from .model import Model, simulate
from .env import bipedhard_custom, Env_config
from collections import OrderedDict
logger = logging.getLogger(__name__)

START_THRESHOLD = 8
DEATH_MARGIN = 5

DEFAULT_ENV = Env_config(
        name='default_env',
        ground_roughness=0,
        pit_gap=[],
        stump_width=[],
        stump_height=[],
        stump_float=[],
        stair_height=[],
        stair_width=[],
        stair_steps=[])


class Box2DNiche(Niche):
    def __init__(self, env_configs, env_params, seed, init='random', stochastic=False, img_creator=None):
        self.model = Model(bipedhard_custom)
        if not isinstance(env_configs, list):
            env_configs = [env_configs]
        self.env_configs = OrderedDict()
        for env in env_configs:
            self.env_configs[env.name] = env
        self.env_params = env_params
        self.seed = seed
        self.stochastic = stochastic
        self.model.make_env(seed=seed, env_config=DEFAULT_ENV)
        self.init = init
        self.img_creator = img_creator

    def __getstate__(self):
        return {"env_configs": self.env_configs,
                "env_params": self.env_params,
                "seed": self.seed,
                "stochastic": self.stochastic,
                "init": self.init,
                "img_file_location": self.img_creator.dataset_folder if self.img_creator is not None else None,
                }

    def __setstate__(self, state):
        from obstacle_detector.niche_image_creator import NicheImageCreator
        self.model = Model(bipedhard_custom)
        self.env_configs = state["env_configs"]
        self.env_params = state["env_params"]
        self.seed = state["seed"]
        self.stochastic = state["stochastic"]
        self.model.make_env(seed=self.seed, env_config=DEFAULT_ENV)
        self.init = state["init"]
        if state["img_file_location"] is not None:
            self.img_creator = NicheImageCreator(cppn_params=self.env_params, dataset_folder=state["img_file_location"])
        else:
            self.img_creator = None

    def add_env(self, env):
        env_name = env.name
        assert env_name not in self.env_configs.keys()
        self.env_configs[env_name] = env

    def delete_env(self, env_name):
        assert env_name in self.env_configs.keys()
        self.env_configs.pop(env_name)

    def initial_theta(self):
        if self.init == 'random':
            return self.model.get_random_model_params()
        elif self.init == 'zeros':
            import numpy as np
            return np.zeros(self.model.param_count)
        else:
            raise NotImplementedError(
                'Undefined initialization scheme `{}`'.format(self.init))

    def rollout(self, theta, random_state, evaluate=False, render_mode=False, gather_obstacle_dataset=False):
        self.model.set_model_params(theta)
        total_returns = 0
        total_length = 0
        if self.stochastic:
            seed = random_state.randint(1000000)
        else:
            seed = self.seed
        for env_config in self.env_configs.values():
            returns, lengths, info = simulate(self.model, seed=seed, train_mode=not evaluate, render_mode=render_mode,
                                              num_episode=1, env_config_this_sim=env_config, env_params=self.env_params)
            total_returns += returns[0]
            total_length += lengths[0]

        if gather_obstacle_dataset and self.img_creator is not None:
            # The function should save dataset images from runtime
            # If the position is significantly far from the start point,
            # and the env is not the initial, flat terrain, draw dataset images
            if info['pos'].x > START_THRESHOLD and self.env_params.cppn_genome.key != "0":
                if info['game_over']:
                    # if the bot died, draw the stumble position
                    label = 'obstacle'
                    self.img_creator.cppn_params = self.env_params
                    image_of_obstacle = self.img_creator.create_image(mid_x=info['pos'].x,
                                                                      in_width=8, in_height=8,
                                                                      out_width=32, out_height=32)
                    self.img_creator.save_image_for_dataset(image_to_save=image_of_obstacle,
                                                            class_label=label,
                                                            x_pos=info['pos'].x,
                                                            cppn_key=self.env_params.cppn_genome.key)

                # For each spot labeled as non_stumbles (which could be none),
                # check that it is significantly far from end/stumble_position, then draw it
                label = 'non_obstacle'
                for position in info['non_stumble_positions']:
                    if position.x < info['pos'].x - DEATH_MARGIN:
                        self.img_creator.cppn_params = self.env_params
                        image_of_non_obstacle = self.img_creator.create_image(mid_x=position.x,
                                                                              in_width=8, in_height=8,
                                                                              out_width=32, out_height=32)
                        self.img_creator.save_image_for_dataset(image_to_save=image_of_non_obstacle, class_label=label,
                                                                x_pos=position.x,
                                                                cppn_key=self.env_params.cppn_genome.key)

        return total_returns / len(self.env_configs), total_length

