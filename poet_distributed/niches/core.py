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
from abc import abstractmethod


class Niche:
    def rollout_batch(self, thetas, batch_size, random_state, eval=False, gather_obstacle_dataset=False):
        import numpy as np
        returns = np.zeros(batch_size)
        lengths = np.zeros(batch_size, dtype='int')
        end_x_positions = np.zeros(batch_size)
        results = np.zeros(batch_size, dtype='bool')

        for i, theta in enumerate(thetas):
            returns[i], lengths[i], end_x_positions[i], results[i] = self.rollout(
                theta, random_state=random_state,
                evaluate=eval, gather_obstacle_dataset=gather_obstacle_dataset
            )

        return returns, lengths, end_x_positions, results

    @abstractmethod
    def rollout(self, theta, random_state, evaluate=False, render_mode=False, gather_obstacle_dataset=False):
        raise NotImplementedError
