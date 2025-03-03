import logging
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
import torch
import torch.nn as nn
from tstf.algo_tstf.utils import orthogonal_init

logger = logging.getLogger(__name__)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_size=256):
        nn.Module.__init__(self)
        self.hiddens = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh()
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        orthogonal_init(self.value[0], 0.01)

    def forward(self, x):
        x = self.hiddens(x)
        value = torch.reshape(self.value(x), [-1])
        return value
