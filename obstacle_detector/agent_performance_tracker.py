import bisect
import numpy as np
from statistics import mean
import tensorflow as tf
from tensorflow.keras import layers, models

SCORE_MAX = 350
SCORE_MIN = -50
SCORE_DIFF = SCORE_MAX - SCORE_MIN

POS_MAX = 90
POS_MIN = 0
POS_DIFF = POS_MAX - POS_MIN


def interpolate_performance_updates(obstacle_classes, obstacle_positions, end_positions, results, success_reward=0.5):
    """
    Calculates how to update each obstacle tracked by the AgentPerformanceTracker, based on the outcome of
    a series of simulations.

    Parameters
    ----------
    obstacle_classes    :   list of int,
                            list of class labels of obstacles in an environment
    obstacle_positions  :   list of float,
                            list of positions of obstacles in an environment
    end_positions       :   list of float,
                            list of positions an agent were at the end of each of a series of simulations
    results             :   list of bool,
                            bool list representing the outcomes of the "end_positions"-simulations.
                            True means the simulation ended in success, false means it ended with the agent dying.
    success_reward      :   float, default=0.5,
                            A factor to scale how much one successful traversal of an obstacle gives as certainty in
                            it's ability to do it again.

    Returns
    -------
    performance_updates : list of (int, float)
    """
    performance_updates = []
    for pos, result in zip(end_positions, results):
        index_before, index_after = get_index_before_and_after(pos, obstacle_positions)

        index_before = max(0, index_before)
        index_before = min(len(obstacle_positions) - 2, index_before)

        index_after = min(len(obstacle_positions) - 1, index_after)
        index_after = max(1, index_after)

        if not result:
            # The agent failed on this position
            # We update the closest obstacles as failures, weighted by proximity.
            pos_before, class_of_pos_before = obstacle_positions[index_before], obstacle_classes[index_before]
            pos_after, class_of_pos_after = obstacle_positions[index_after], obstacle_classes[index_after]

            # print(f"Pos_bef {pos_before}, Pos_aft {pos_after}, Index bef {index_before}, Index aft {index_after}")
            weight_before = (pos - pos_before)/(pos_after - pos_before)
            weight_after = 1 - weight_before

            performance_updates.append((class_of_pos_before, -weight_before))
            performance_updates.append((class_of_pos_after, -weight_after))

        # All locations up to the pos should be considered a success.
        for i in range(index_before):
            performance_updates.append((obstacle_classes[i], success_reward))

    return performance_updates


def get_index_before_and_after(pos, position_list):
    """
    Finds the index of the positions before and after the given position
    Parameters
    ----------
    pos    : int
    position_list   : list of int

    Returns
    -------
    (int, int)
    """
    index1 = bisect.bisect_left(position_list, pos)
    index0 = index1-1
    return index0, index1


class AgentPerformanceTracker:

    def __init__(self, certainty_threshold=0.8, learning_rate=0.1, n_pos_score_limit=1000):
        """
        Tracks the performance of an agent against the classes of obstacles

        Parameters
        ----------
        certainty_threshold
        learning_rate
        """
        self._certainty_threshold = certainty_threshold
        self._learning_rate = learning_rate
        self.obstacle_performance = {}

        self.model = models.Sequential()
        self.model.add(layers.Dense(units=3, activation='relu', input_dim=1))
        self.model.add(layers.Dense(units=3, activation='relu'))
        self.model.add(layers.Dense(units=3, activation='relu'))
        self.model.add(layers.Dense(units=1))

        self.model.compile(
            optimizer='sgd',
            loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
            metrics=['accuracy']
        )

        # We don't use this any more
        # self.positions = []
        # self.scores = []
        # self.n_pos_score_limit = n_pos_score_limit

    def get_class_performance(self, obstacle_class_id):
        if obstacle_class_id in self.obstacle_performance:
            return self.obstacle_performance[obstacle_class_id]
        else:
            self.obstacle_performance[obstacle_class_id] = 0
            return self.obstacle_performance[obstacle_class_id]

    def update_class_performance(self, performance_updates):
        """
        Updates expected class performance based on the performance against a recent instance
        Parameters
        ----------
        performance_updates     :   list of (int, float)
                                    list of (Class ID, score) tuples to use for updating the class performance of the
                                    agent.
        Returns
        -------
        None
        """
        for obstacle_class_id, performance_score in performance_updates:
            new_score = self.get_class_performance(obstacle_class_id)

            new_score += self._learning_rate*performance_score
            new_score = 1 if new_score > 1 else -1 if new_score < -1 else new_score  # clamp between -1 and 1

            self.obstacle_performance[obstacle_class_id] = new_score

    def predict_class_performance(self, obstacle_class_id):
        """

        Parameters
        ----------
        obstacle_class_id   :   int
                                ID of the obstacle class to be updated.

        Returns
        -------
        int :   1 if success is predicted, -1 if failure is predicted, 0 if uncertain
        """
        class_performance = self.get_class_performance(obstacle_class_id)
        # print(class_performance, self._certainty_threshold)
        if class_performance > self._certainty_threshold:
            prediction = 1
        elif class_performance < -self._certainty_threshold:
            prediction = -1
        else:
            prediction = 0

        return prediction

    def predict_simulation_distance(self, obstacle_classes, obstacle_positions):
        for ob_class, ob_pos in zip(obstacle_classes, obstacle_positions):
            ob_performance = self.predict_class_performance(ob_class)
            # print(ob_performance)
            if ob_performance == 1:
                continue
            elif ob_performance == -1:
                return ob_pos, ob_performance
            else:
                return None, ob_performance

        # Predicted full success!
        return ob_pos, ob_performance

    # def update_position_score_relation(self, positions, scores):
    #     for pos, score in zip(positions, scores):
    #         self.positions.append(pos), self.scores.append(score)
    #         if len(self.positions) >= self.n_pos_score_limit:
    #             self.positions.pop(0), self.scores.pop(0)
    #
    # def get_position_score_relation(self):
    #     relation = np.mean(np.array(self.scores) / np.array(self.positions))
    #     return relation

    def train_simulation_score(self, positions, scores):
        X = (np.array(positions) - POS_MIN)/POS_DIFF
        y = (np.array(scores) - SCORE_MIN)/SCORE_DIFF
        self.model.fit(x=X, y=y)

    def predict_simulation_score(self, position):
        position = (position - POS_MIN)/POS_DIFF
        input = [position]
        score = self.model.predict(input)[0][0]
        return (score * SCORE_DIFF) + SCORE_MIN


if __name__ == '__main__':
    performance_updates = [(i % 10, 1) for i in range(100)]
    end_x_pos = np.array([10 for i in range(100000)])
    score = np.array([100 for i in range(100000)])

    apt = AgentPerformanceTracker(certainty_threshold=0.0)
    apt.update_class_performance(performance_updates)
    apt.train_simulation_score(end_x_pos, score)
    print(apt.predict_simulation_score(10))

