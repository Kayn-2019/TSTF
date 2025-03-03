import torch
import torch.nn as nn
import gym
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from tstf.algo_tstf.actor import Actor
from tstf.algo_tstf.critic import Critic
from collections import deque
from gym.spaces import Box
from tstf.algo_tstf.simsiam import SimSiam
from ray.rllib.policy.view_requirement import ViewRequirement
from tstf.algo_tstf.utils import orthogonal_init
from tstf.algo_tstf.gated_transformer import TransformerEncoder
from tstf.algo_tstf.augment import augment_sample


class CustomDeque:
    def __init__(self, maxlen):
        self.queue = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def enqueue(self, element):
        if len(self.queue) < self.maxlen:
            head_element = self.queue[0] if self.queue else element
            while len(self.queue) < self.maxlen:
                self.queue.appendleft(head_element)
        else:
            self.queue.append(element)

    def clear(self):
        self.queue.clear()


class TSTFModel(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        dim_obs = np.product(obs_space.shape)
        # dim_obs = 19
        dim_action = np.product(action_space.shape)
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.dim_encoder = 256
        self.dim_feature = 256
        self.max_time_step = model_config["custom_model_config"]["max_time_step"]
        self.max_num_agent = model_config["custom_model_config"]["max_num_agent"]
        self.obs_fc = nn.Sequential(
            nn.Linear(self.dim_obs, self.dim_feature),
            nn.Tanh(),
        )
        orthogonal_init(self.obs_fc[0])
        self.temporal_fc = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_feature),
            nn.Tanh(),
        )
        orthogonal_init(self.temporal_fc[0])
        self.spacial_fc = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_feature),
            nn.Tanh()
        )
        orthogonal_init(self.spacial_fc[0])
        self.temporal_encoder = TransformerEncoder(21)
        self.temporal_simsiam = SimSiam(self.temporal_encoder)
        self.spacial_encoder = TransformerEncoder(21)
        self.spacial_simsiam = SimSiam(self.spacial_encoder)
        self.actor = Actor(self.dim_feature, self.dim_action)
        self.critic = Critic(self.dim_feature)
        self.custom_deque = CustomDeque(self.max_time_step)
        self.view_requirements["obs"] = ViewRequirement(
            space=Box(obs_space.low[0], obs_space.high[0], shape=((dim_obs+2) * self.max_num_agent,))
        )

    def get_temporal_feature(self, input_dict):
        region_obs = input_dict["obs"].float()
        obs = region_obs[:,:self.dim_obs]
        if self.training:
            obs_history = input_dict["obs_history"].float()
        else:
            if self.custom_deque.queue and obs.shape[0] != self.custom_deque.queue[0].shape[0]:
                self.custom_deque.clear()
            self.custom_deque.enqueue(obs)
            queue_list = list(self.custom_deque.queue)
            obs_history = torch.stack(queue_list).permute(1, 0, 2)
        temporal_indices = list(range(19)) + [-2, -1]
        obs_history = obs_history[:, :, temporal_indices]
        feature = self.temporal_encoder(obs_history)
        return feature

    def get_spacial_feature(self, input_dict):
        region_obs = input_dict["obs"].float()
        region_obs = torch.reshape(region_obs,(-1, self.max_num_agent, self.dim_obs+2))
        spacial_indices = list(range(19))+ [-2, -1]
        region_obs = region_obs[:, :, spacial_indices]
        region_obs = torch.flip(region_obs, [1])
        feature = self.spacial_encoder(region_obs)
        return feature

    def forward(self, input_dict, state, seq_lens):
        region_obs = input_dict["obs"].float()
        obs = region_obs[:,:self.dim_obs]
        obs_encoded = self.obs_fc(obs)
        temporal_feature = self.get_temporal_feature(input_dict).detach()
        temporal_feature = self.temporal_fc(temporal_feature)
        spacial_feature = self.get_spacial_feature(input_dict).detach()
        spacial_feature = self.spacial_fc(spacial_feature)
        # actor_in = obs_encoded + temporal_feature + spacial_feature
        actor_in = obs_encoded + temporal_feature + spacial_feature
        logits = self.actor(actor_in)
        return logits, state

    def value_function(self):
        raise ValueError(
            "Centralized Value Function should not be called directly! "
            "Call central_value_function(state) instead!"
        )

    def central_value_function(self, input_dict):
        region_obs = input_dict["obs"].float()
        obs = region_obs[:, :self.dim_obs]
        obs_encoded = self.obs_fc(obs)
        temporal_feature = self.get_temporal_feature(input_dict).detach()
        temporal_feature = self.temporal_fc(temporal_feature)
        spacial_feature = self.get_spacial_feature(input_dict).detach()
        spacial_feature = self.spacial_fc(spacial_feature)
        # critic_in = obs_encoded + temporal_feature + spacial_feature
        critic_in = obs_encoded + temporal_feature + spacial_feature
        value = self.critic(critic_in)
        return value

    def temporal_simsiam_loss(self, input_dict):
        criterion = nn.CosineSimilarity(dim=1)
        temporal_indices = list(range(19))+ [-2, -1]
        obs_history = input_dict["obs_history"][:, :, temporal_indices].float()
        obs_history_aug = augment_sample(obs_history)
        p1, p2, z1, z2 = self.temporal_simsiam(x1=obs_history, x2=obs_history_aug)
        loss = 1 - (criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        return loss

    def spacial_simsiam_loss(self, input_dict):
        criterion = nn.CosineSimilarity(dim=1)
        region_obs = input_dict["obs"].float()
        region_obs = torch.reshape(region_obs,(-1, self.max_num_agent, self.dim_obs+2))
        spacial_indices = list(range(19))+ [-2, -1]
        region_obs = region_obs[:, :, spacial_indices]
        region_obs = torch.flip(region_obs, [1])
        region_obs_aug = augment_sample(region_obs)
        p1, p2, z1, z2 = self.spacial_simsiam(x1=region_obs, x2=region_obs_aug)
        loss = 1 - (criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        return loss
