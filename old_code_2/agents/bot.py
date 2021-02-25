import torch
from columns_gym import envs
from exploration_strategies import Base_Exploration_Strategy


class BaseAgent(object):
    def __init__(self, estimator: torch.nn.Module):
        """
        BaseAgent initialization.

        Parameters:

            estimator: torch.nn.Module
                Neural network instance.

        """

        self.estimator = estimator

    def take_action(self, env_state) -> None:
        raise NotImplementedError


# TODO implement
class RandomAgent(object):

    # TODO
    def take_action(self, envs_state):
        """Choose random action depends on env_state."""
        while envs_state.is_not_over:
            envs_state.state, envs_state.reward = envs_state.envs.step()
        return envs_state.fin_sentense


# TODO implement
class QAgent(BaseAgent):
    def __init__(self, estimator: torch.nn.Module, epsilon: float = 0.0):
        super().__init__(estimator)

        self.epsilon = epsilon

    def take_action(self, env_state):
        pass

    def update_estimator(self, estimator):
        self.estimator = estimator

    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon
