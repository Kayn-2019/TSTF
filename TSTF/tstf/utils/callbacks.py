from collections import defaultdict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from typing import TYPE_CHECKING, Dict, Optional, Union
import numpy as np
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker


class MultiAgentDrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        episode.user_data["velocity"] = defaultdict(list)
        episode.user_data["steering"] = defaultdict(list)
        episode.user_data["step_reward"] = defaultdict(list)
        episode.user_data["acceleration"] = defaultdict(list)
        episode.user_data["cost"] = defaultdict(list)
        episode.user_data["episode_length"] = defaultdict(list)
        episode.user_data["episode_reward"] = defaultdict(list)
        episode.user_data["num_neighbours"] = defaultdict(list)

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        active_keys = list(base_env.envs[env_index].vehicles.keys())

        # The agent_rewards dict contains all agents' reward, not only the active agent!
        # active_keys = [k for k, _ in episode.agent_rewards.keys()]

        for agent_id in active_keys:
            k = agent_id
            info = episode.last_info_for(k)
            if info:
                if "step_reward" not in info:
                    continue
                episode.user_data["velocity"][k].append(info["velocity"])
                episode.user_data["steering"][k].append(info["steering"])
                episode.user_data["step_reward"][k].append(info["step_reward"])
                episode.user_data["acceleration"][k].append(info["acceleration"])
                episode.user_data["cost"][k].append(info["cost"])
                episode.user_data["episode_length"][k].append(info["episode_length"])
                episode.user_data["episode_reward"][k].append(info["episode_reward"])
                episode.user_data["num_neighbours"][k].append(len(info.get("neighbours", [])))

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        **kwargs
    ):
        keys = [k for k, _ in episode.agent_rewards.keys()]
        arrive_dest_list = []
        crash_list = []
        out_of_road_list = []
        max_step_rate_list = []

        # Newly introduced metrics
        track_length_list = []
        route_completion_list = []
        current_distance_list = []

        for k in keys:
            info = episode.last_info_for(k)
            arrive_dest = info.get("arrive_dest", False)

            # Newly introduced metrics
            route_completion = info.get("route_completion", -1)
            track_length = info.get("track_length", -1)
            current_distance = info.get("current_distance", -1)
            track_length_list.append(track_length)
            current_distance_list.append(current_distance)
            route_completion_list.append(route_completion)

            crash = info.get("crash", False)
            out_of_road = info.get("out_of_road", False)
            max_step_rate = not (arrive_dest or crash or out_of_road)
            arrive_dest_list.append(arrive_dest)
            crash_list.append(crash)
            out_of_road_list.append(out_of_road)
            max_step_rate_list.append(max_step_rate)

        # Newly introduced metrics
        episode.custom_metrics["track_length"] = np.mean(track_length_list)
        episode.custom_metrics["current_distance"] = np.mean(current_distance_list)
        episode.custom_metrics["route_completion"] = np.mean(route_completion_list)

        episode.custom_metrics["success_rate"] = np.mean(arrive_dest_list)
        episode.custom_metrics["crash_rate"] = np.mean(crash_list)
        episode.custom_metrics["out_of_road_rate"] = np.mean(out_of_road_list)
        episode.custom_metrics["max_step_rate"] = np.mean(max_step_rate_list)

        for info_k, info_dict in episode.user_data.items():
            self._add_item(episode, info_k, [vv for v in info_dict.values() for vv in v])

        agent_cost_list = [sum(episode_costs) for episode_costs in episode.user_data["cost"].values()]
        episode.custom_metrics["episode_cost"] = np.mean(agent_cost_list)
        episode.custom_metrics["episode_cost_worst_agent"] = np.min(agent_cost_list)
        episode.custom_metrics["episode_cost_best_agent"] = np.max(agent_cost_list)
        episode.custom_metrics["environment_cost_total"] = np.sum(agent_cost_list)
        episode.custom_metrics["num_active_agents"] = len(agent_cost_list)
        episode.custom_metrics["episode_length"] = np.mean(
            [ep_len[-1] for ep_len in episode.user_data["episode_length"].values()]
        )
        episode.custom_metrics["episode_reward"] = np.mean(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        )
        episode.custom_metrics["environment_reward_total"] = np.sum(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        )

    def _add_item(self, episode, name, value_list):
        # episode.custom_metrics["{}_max".format(name)] = float(np.max(value_list))
        episode.custom_metrics["{}".format(name)] = float(np.mean(value_list))
        # episode.custom_metrics["{}_min".format(name)] = float(np.min(value_list))
        # pass

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["rc"] = np.nan
        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]

        if "route_completion_mean" in result["custom_metrics"]:
            result["rc"] = result["custom_metrics"]["route_completion_mean"]

        result["cost"] = np.nan
        if "episode_cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["episode_cost_mean"]

        # present the agent-averaged reward.
        result["raw_episode_reward_mean"] = result["episode_reward_mean"]

        # Fill Per agent reward as the item "episode_reward_mean", instead of the summation.
        policy_reward_mean = list(result["policy_reward_mean"].values())
        if len(policy_reward_mean) == 0:
            if "episode_reward_mean" in result["custom_metrics"]:
                result["episode_reward_mean"] = result["custom_metrics"]["episode_reward_mean"]
        else:
            result["episode_reward_mean"] = np.mean(policy_reward_mean)
        # result["environment_reward_total"] = np.sum(list(result["policy_reward_mean"].values()))

    # def on_sample_end(self, *, worker, samples, **kwargs):
    #     print("===== Sample is ended! =====\n{} agent steps and count is {}, {} count by {}".format(
    #         samples.agent_steps(), samples.count, samples.env_steps(), worker.config.count_steps_by
    #     ))
