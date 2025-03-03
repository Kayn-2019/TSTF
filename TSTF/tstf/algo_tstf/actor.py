import logging
import torch
import torch.nn as nn
from tstf.algo_tstf.utils import orthogonal_init

logger = logging.getLogger(__name__)


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=256):
        nn.Module.__init__(self)
        self.hiddens = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh()
        )
        orthogonal_init(self.hiddens[0])
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
        )
        orthogonal_init(self.mean[0], gain=0.01)
        self.std_log = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
        )
        orthogonal_init(self.std_log[0], gain=0.01)

    def forward(self, x):
        x = self.hiddens(x)
        mean = self.mean(x)
        std_log = self.std_log(x)
        logits = torch.cat([mean, std_log], dim=-1)
        return logits
