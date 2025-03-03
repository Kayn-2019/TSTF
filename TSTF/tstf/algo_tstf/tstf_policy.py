import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
)
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.numpy import convert_to_numpy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.evaluation.postprocessing import compute_advantages, discount_cumsum
from ray.rllib.utils.typing import LocalOptimizer


class TSTFPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.simsiam_terminate = False
        super().__init__(observation_space, action_space, config)

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {}

    def loss(self, model, dist_class, train_batch):

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP]
        )

        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()

        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] *
            torch.clamp(logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]),
        )

        # Compute a value function loss.
        assert self.config["use_critic"]

        value_fn_out = model.central_value_function(train_batch)
        if torch.isnan(value_fn_out).any():
            raise ValueError("The value_fn_out contains NaN values.")

        if self.config["old_value_loss"]:
            current_vf = value_fn_out
            prev_vf = train_batch[SampleBatch.VF_PREDS]
            vf_loss1 = torch.pow(current_vf - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_clipped = prev_vf + torch.clamp(
                current_vf - prev_vf, -self.config["vf_clip_param"], self.config["vf_clip_param"]
            )
            vf_loss2 = torch.pow(vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.max(vf_loss1, vf_loss2)
        else:
            vf_loss = torch.pow(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)

        total_loss = reduce_mean_valid(
            -surrogate_loss + self.config["vf_loss_coeff"] * vf_loss_clipped - self.entropy_coeff * curr_entropy
        )

        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        temporal_simsiam_loss = self.model.temporal_simsiam_loss(train_batch)
        spacial_simsiam_loss = self.model.spacial_simsiam_loss(train_batch)
        # total_loss = total_loss + spacial_simsiam_loss + temporal_simsiam_loss

        model.tower_stats["temporal_simsiam_loss"] = temporal_simsiam_loss
        model.tower_stats["spacial_simsiam_loss"] = spacial_simsiam_loss
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["vf"] = value_fn_out
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return [temporal_simsiam_loss, spacial_simsiam_loss, total_loss]

    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "temporal_simsiam_loss": torch.mean(
                    torch.stack(self.get_tower_stats("temporal_simsiam_loss"))
                ),
                "spacial_simsiam_loss": torch.mean(
                    torch.stack(self.get_tower_stats("spacial_simsiam_loss"))
                ),

                "vf": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    def region_process(self, sample_batch, other_agent_batches):
        dim_obs = self.model.dim_obs
        ego_obs = sample_batch["obs"][:, 0:dim_obs]
        ego_action = sample_batch["actions"]
        ego_obs_action = np.concatenate((ego_obs, ego_action), axis=-1)
        obs_history = []
        action_history = []
        obs_action_history = []
        time_step = self.config["max_time_step"]
        for i in range(len(ego_obs_action)):
            start_idx = max(0, i - time_step + 1)
            multi_step_obs = ego_obs[start_idx:i + 1]
            multi_step_act = ego_action[start_idx:i + 1]
            multi_step_obs_act = ego_obs_action[start_idx:i + 1]
            if i < time_step:
                obs_pad = np.repeat(multi_step_obs[:1, :], time_step - i - 1, axis=0)
                act_pad = np.repeat(multi_step_act[:1, :], time_step - i - 1, axis=0)
                obs_act_pad = np.repeat(multi_step_obs_act[:1, :], time_step - i - 1, axis=0)
                multi_step_obs = np.vstack([obs_pad, multi_step_obs])
                multi_step_act = np.vstack([act_pad, multi_step_act])
                multi_step_obs_act = np.vstack([obs_act_pad, multi_step_obs_act])
            obs_history.append(multi_step_obs)
            action_history.append(multi_step_act)
            obs_action_history.append(multi_step_obs_act)

        sample_batch["obs_history"] = np.array(obs_history)
        sample_batch["action_history"] = np.array(action_history)
        sample_batch["obs_action_history"] = np.array(obs_action_history)

        # region_obs = []
        # region_rewards = []
        # region_num_agent = []
        # for index in range(sample_batch.count):
        #     region_obs_step = []
        #     region_rewards_step = []
        #     ego_obs_step = ego_obs[index]
        #     ego_action_step = ego_action[index]
        #     # region_obs_step.append(np.concatenate((ego_obs_step, ego_action_step),axis=-1))
        #     region_obs_step.append(ego_obs_step)
        #     ego_reward = sample_batch["rewards"][index]
        #     region_rewards_step.append(ego_reward)
        #     environmental_time_step = sample_batch["t"][index]
        #     neighbours = sample_batch['infos'][index]["neighbours"]
        #     neighbours_distance = sample_batch['infos'][index]["neighbours_distance"]
        #     obs_list = []
        #     act_list = []
        #     for nei_count, (nei_name, nei_dist) in enumerate(zip(neighbours, neighbours_distance)):
        #         if nei_count >= self.config["max_num_agent"] - 1:
        #             break
        #
        #         nei_act = None
        #         nei_obs = None
        #         nei_reward = None
        #         if nei_name in other_agent_batches:
        #             _, nei_batch = other_agent_batches[nei_name]
        #
        #             match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]
        #
        #             if len(match_its_step) == 0:
        #                 pass
        #             elif len(match_its_step) > 1:
        #                 raise ValueError()
        #             else:
        #                 new_index = match_its_step[0]
        #                 nei_obs = nei_batch[SampleBatch.CUR_OBS][new_index][0:dim_obs]
        #                 nei_act = nei_batch[SampleBatch.ACTIONS][new_index]
        #                 nei_reward = nei_batch[SampleBatch.REWARDS][new_index]
        #
        #         if nei_obs is not None:
        #             obs_list.append(nei_obs)
        #             act_list.append(nei_act)
        #             # region_obs_step.append(np.concatenate([nei_obs, nei_act]))
        #             region_obs_step.append(nei_obs)
        #             region_rewards_step.append(nei_reward)
        #
        #     region_num_agent.append(len(region_obs_step))
        #     if len(region_obs_step) < self.config["max_num_agent"]:
        #         num_pad = self.config["max_num_agent"] - len(region_obs_step)
        #         for _ in range(num_pad):
        #             # region_obs_step.append(np.zeros(len(ego_obs_step)+len(ego_action_step)))
        #             region_obs_step.append(np.zeros(len(ego_obs_step)))
        #             region_rewards_step.append(0)
        #
        #     region_obs.append(region_obs_step)
        #     region_rewards.append(region_rewards_step)
        #
        # sample_batch["region_num_agent"] = np.array(region_num_agent)
        # sample_batch["region_obs"] = np.array(region_obs)
        # sample_batch["region_rewards_list"] = np.array(region_rewards)

        return sample_batch

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        with torch.no_grad():
            if episode is not None:
                self.region_process(sample_batch, other_agent_batches)
            else:
                # dim_state_obs = self.model.dim_obs
                dim_obs = self.model.dim_obs
                sample_batch["region_obs"] = np.zeros((sample_batch.count, self.config["max_num_agent"], dim_obs))
                sample_batch["obs_history"] = np.zeros((sample_batch.count, self.config["max_time_step"], dim_obs))
            sample_batch[SampleBatch.VF_PREDS] = self.model.central_value_function(
                convert_to_torch_tensor(sample_batch, self.device)
            ).cpu().detach().numpy().astype(np.float32)

            if sample_batch[SampleBatch.DONES][-1]:
                last_r = 0.0
            else:
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]
            sample_batch = compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True)
            )

        return sample_batch

    def optimizer(self, ):
        temporal_simsiam_optimizer = torch.optim.Adam(
            params=list(self.model.temporal_simsiam.parameters()),
            lr=1e-3,
            eps=1e-7,
        )
        spacial_simsiam_optimizer = torch.optim.Adam(
            params=list(self.model.spacial_simsiam.parameters()),
            lr=1e-3,
            eps=1e-7,
        )
        optimizer = torch.optim.Adam(
            params=list(self.model.obs_fc.parameters()) + list(self.model.temporal_fc.parameters())
                   + list(self.model.spacial_fc.parameters()) + list(self.model.actor.parameters()) + list(self.model.critic.parameters()),
            lr=1e-3,
            eps=1e-7,
        )
        return [temporal_simsiam_optimizer, spacial_simsiam_optimizer, optimizer]



