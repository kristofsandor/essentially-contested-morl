from .fire_rescue import Actions, FireRescueEnv
import numpy as np


class TaskIdx(Enum):
    DIAMOND = 0
    RESCUE = 1


class FireRescueBaseline(FireRescueEnv):

    def step(self, action: int):
        reward_vector = np.zeros(2)
        self.step_count += 1

        diamond_collected = False

        if action == Actions.RESCUE.value:
            self._handle_rescue()
        elif action == Actions.COLLECT.value:
            diamond_collected = self._handle_collect()
        else:  # Movement
            self._handle_movement(action)

        if diamond_collected:
            reward_vector[TaskIdx.DIAMOND.value] += 1
