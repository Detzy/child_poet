

class AgentPerformanceTracker:

    def __init__(self, certainty_threshold=0.8, learning_rate=0.1):
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

    def update_class_performance(self, obstacle_class_id, beat_obstacle):
        """
        Updates expected class performance based on the performance against a recent instance
        Parameters
        ----------
        obstacle_class_id   :   int
                                ID of the obstacle class to be updated.
        beat_obstacle       :   bool
                                True if the obstacle was overcome by the agent, false if it killed the agent.
        Returns
        -------
        None
        """
        performance_score = 1 if beat_obstacle else -1
        new_score = self.obstacle_performance[obstacle_class_id]
        new_score += self._learning_rate*performance_score
        new_score = 1 if new_score > 1 else -1 if new_score < -1 else new_score  # clamp between -1 and 1
        self.obstacle_performance[obstacle_class_id] = new_score

    def get_class_performance(self, obstacle_class_id):
        return self.obstacle_performance[obstacle_class_id]

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
        if self.obstacle_performance[obstacle_class_id] > self._certainty_threshold:
            prediction = 1
        elif self.obstacle_performance[obstacle_class_id] < -self._certainty_threshold:
            prediction = -1
        else:
            prediction = 0

        return prediction
