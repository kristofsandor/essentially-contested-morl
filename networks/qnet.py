import torch
import torch.nn as nn

from networks.cnn import CNN
from morl_baselines.common.networks import mlp, layer_init


class CNNQNet(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        reward_dim,
        net_arch,
        cnn_config,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.rew_dim = reward_dim
        self.feature_extractor = CNN(
            obs_shape=obs_shape,
            **cnn_config,
        )

        self.net = mlp(
            cnn_config.get("out_features") + reward_dim, action_dim * reward_dim, net_arch
        )
        self.apply(layer_init)

    def forward(self, obs, w):
        features = self.feature_extractor(obs)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        x = torch.cat((features, w), dim=features.dim() - 1)

        q_values = self.net(x)

        return q_values.view(-1, self.action_dim, self.rew_dim)
