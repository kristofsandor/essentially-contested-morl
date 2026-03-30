import gymansium as gym

class ResourceGatherWrapper(gym.RewardWrapper):

    def __init__(self, env)  -> None:
        super().__init__(env)

    def 