from ray.rllib.algorithms.ppo.ppo import PPOConfig
import gym
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.algorithm import Algorithm
from tstf.algo_tstf.tstf_policy import TSTFPolicy
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.rollout_ops import standardize_fields
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
import logging
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from typing import Type
from ray.util.debug import log_once
from tstf.algo_tstf.tstf_model import TSTFModel
from ray.rllib.models.catalog import ModelCatalog
from metadrive.engine.base_engine import BaseEngine
from typing import Dict

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import AgentID, PolicyID
import numpy as np

ModelCatalog.register_custom_model("tstf_model", TSTFModel)
logger = logging.getLogger(__name__)


def preprocess_obs(
        agent_obs: Dict[AgentID, TensorType],
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kw
) -> Dict[AgentID, TensorType]:
    assert agent_obs is not None
    processed_obs = {}
    num_agents = 5
    for agent_id, obs in agent_obs.items():
        info = episode.last_info_for(agent_id)
        region_obs = []
        region_obs.append(obs)
        if info is not None and (len(info) != 0):
            for i, nei_name in enumerate(info["neighbours"]):
                if i >= num_agents - 1:
                    break
                region_obs.append(agent_obs[nei_name])
        if len(region_obs) < num_agents:
            num_pad = num_agents - len(region_obs)
            for _ in range(num_pad):
                region_obs.append(np.zeros(len(obs), dtype='float32'))
        region_obs_np = np.concatenate(region_obs)
        processed_obs[agent_id] = region_obs_np

    return processed_obs


class TSTFConfig(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or TSTF)
        self.sgd_minibatch_size = 512
        self.train_batch_size = 2000
        self.num_sgd_iter = 5
        self.lr_schedule = [(0, 1e-3), (600000, 1e-5)]
        self.clip_param = 0.3
        self.lambda_ = 0.95
        self.num_cpus_per_worker = 0.2
        self.num_cpus_for_local_worker = 1
        self.num_rollout_workers = 8
        self.framework_str = "torch"
        self.vf_clip_param = 100
        self.old_value_loss = True
        self.max_num_grad_updates = 20000
        self.max_time_step = 5
        self.max_num_agent = 5
        self.update_from_dict({"model": {"custom_model": "tstf_model",
                                         "custom_model_config": {
                                             "max_time_step": self.max_time_step,
                                             "max_num_agent": self.max_num_agent,
                                         }}})
        self.preprocessor_pref = None

    def validate(self):
        super().validate()
        if BaseEngine.singleton is None:
            from ray.tune.registry import _global_registry, ENV_CREATOR
            from metadrive.constants import DEFAULT_AGENT

            env_class = _global_registry.get(ENV_CREATOR, self["env"])
            single_env = env_class(self["env_config"])
            # lidar_state_observation = single_env.get_single_observation()
            # dim_state_obs = lidar_state_observation.state_obs.observation_space.shape
            # self.model["custom_model_config"]["dim_state_obs"] = dim_state_obs[0]

            if "agent0" in single_env.observation_space.spaces:
                obs_space = single_env.observation_space["agent0"]
                act_space = single_env.action_space["agent0"]
            else:
                obs_space = single_env.observation_space[DEFAULT_AGENT]
                act_space = single_env.action_space[DEFAULT_AGENT]

            assert isinstance(obs_space, gym.spaces.Box)

            self.update_from_dict(
                {
                    "multiagent": dict(
                        # Note that we have to use "default" because stupid RLLib has bug when
                        # we are using "default_policy" as the Policy ID.
                        policies={"default": PolicySpec(None, obs_space, act_space, {})},
                        policy_mapping_fn=lambda x: "default",
                        observation_fn=preprocess_obs,
                    )
                }
            )


class TSTF(Algorithm):
    @classmethod
    def get_default_config(cls):
        return TSTFConfig()

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        assert config["framework"] == "torch"
        return TSTFPolicy

    def training_step(self):
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config.train_batch_size
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config.train_batch_size
            )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
        cur_ts = self._counters[
            NUM_AGENT_STEPS_SAMPLED
            if self.config.count_steps_by == "agent_steps"
            else NUM_ENV_STEPS_SAMPLED
        ]

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["region_advantages"])
        # Train
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        policies_to_update = list(train_results.keys())

        global_vars = {
            # "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(
                    policies=policies_to_update,
                    global_vars=global_vars,
                )

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                    self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                    log_once("ppo_warned_lr_ratio")
                    and self.config.get("model", {}).get("vf_share_layers")
                    and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                    log_once("ppo_warned_vf_clip")
                    and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results
