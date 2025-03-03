import copy
from collections import defaultdict
from math import cos, sin

import numpy as np
import gymnasium
from metadrive.utils import get_np_random, clip
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env
from metadrive.envs.gym_wrapper import createGymWrapper, gymToGymnasium
from metadrive.envs.marl_envs import MultiAgentMetaDrive
from metadrive.obs.state_obs import LidarStateObservation

from metadrive.obs.observation_base import DummyObservation
from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.env_input_policy import EnvInputPolicy  # 简单的策略，允许手动控制车辆
from panda3d.core import Material

from IPython.display import Image
from metadrive.manager.agent_manager import VehicleAgentManager
from metadrive.policy.idm_policy import IDMPolicy

from metadrive.manager.traffic_manager import PGTrafficManager, TrafficMode
import copy
import logging
from collections import namedtuple
from typing import Dict

import math
import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap

import random


def MixIDMPolicy():
    choice = random.randint(0, 2)
    if choice == 0:
        return IDMConservativePolicy
    elif choice == 1:
        return IDMNormalPolicy
    else:
        return IDMAggressivePolicy



class IDMAggressivePolicy(IDMPolicy):
    NORMAL_SPEED = 30
    MAX_SPEED = 100
    DISTANCE_WANTED = 10.0
    TIME_WANTED = 1.5
    LANE_CHANGE_FREQ = 50
    LANE_CHANGE_SPEED_INCREASE = 10
    SAFE_LANE_CHANGE_DISTANCE = 15
    MAX_LONG_DIST = 30


class IDMNormalPolicy(IDMPolicy):
    NORMAL_SPEED = 25
    MAX_SPEED = 60
    TIME_WANTED = 1.8
    DISTANCE_WANTED = 12.0
    LANE_CHANGE_FREQ = 60
    LANE_CHANGE_SPEED_INCREASE = 8
    SAFE_LANE_CHANGE_DISTANCE = 18
    MAX_LONG_DIST = 35



class IDMConservativePolicy(IDMPolicy):
    NORMAL_SPEED = 20
    MAX_SPEED = 40
    TIME_WANTED = 2
    DISTANCE_WANTED = 15.0
    # LANE_CHANGE_FREQ = 50  # [step]
    LANE_CHANGE_SPEED_INCREASE = 6
    SAFE_LANE_CHANGE_DISTANCE = 20
    MAX_LONG_DIST = 35


class MixedTrafficManager(PGTrafficManager):

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logging.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []
        traffic_density = self.density
        self.respawn_lanes = self._get_available_respawn_lanes(map)
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_vehicles_once(map, traffic_density)
        else:
            raise ValueError("No such mode named {}".format(self.mode))

    def _create_respawn_vehicles(self, map: BaseMap, traffic_density: float):
        vehicle_num = self.engine.global_config["num_traffic_vehicle"]
        lane_num = len(self.respawn_lanes)
        for i in range(vehicle_num):
            vehicle_type = self.random_vehicle_type()
            lane = self.respawn_lanes[i % lane_num]
            lane_idx = lane.index
            long = self.np_random.rand() * lane.length / 2
            traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
            new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
            from metadrive.policy.idm_policy import IDMPolicy
            self.add_policy(new_v.id,  IDMNormalPolicy, new_v, self.generate_seed())
            self._traffic_vehicles.append(new_v)

    def after_step(self, *args, **kwargs):
        """
        Update all traffic vehicles' states,
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if not v.on_lane:
                if self.mode == TrafficMode.Trigger:
                    v_to_remove.append(v)
                elif self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                    v_to_remove.append(v)
                else:
                    raise ValueError("Traffic mode error: {}".format(self.mode))
        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
            if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                lane = self.respawn_lanes[self.np_random.randint(0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
                new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                self.add_policy(new_v.id, IDMNormalPolicy, new_v, self.generate_seed())
                self._traffic_vehicles.append(new_v)

        return dict()


class MixedMultiAgentEnv:
    """
    This class maintains a distance map of all agents and appends the
    neighbours' names and distances into info at each step.
    We should subclass this class to a base environment class.
    """

    @classmethod
    def default_config(cls):
        config = super(MixedMultiAgentEnv, cls).default_config()
        config["neighbours_distance"] = 50
        config["num_traffic_vehicle"] = 0.0
        return config

    def __init__(self, *args, **kwargs):
        super(MixedMultiAgentEnv, self).__init__(*args, **kwargs)
        self.bounding_box = None
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

    def step(self, actions):
        # o, r, d, i = super(MixedMultiAgentEnv, self).step(actions)
        o, r, tm, tc, i = super(MixedMultiAgentEnv, self).step(actions)
        self._update_distance_map(dones=(tm or tc))
        global_reward = sum(r.values()) / len(r.values())
        for kkk in i.keys():
            i[kkk]["all_agents"] = list(i.keys())
            neighbours, nei_distances = self._find_in_range(kkk, self.config["neighbours_distance"])
            i[kkk]["neighbours"] = neighbours
            i[kkk]["neighbours_distance"] = nei_distances
            i[kkk]["global_rewards"] = global_reward

        return o, r, tm, tc, i

    def _post_process_config(self, config):
        from metadrive.manager.spawn_manager import SpawnManager
        config = super(MultiAgentMetaDrive, self)._post_process_config(config)
        ret_config = config
        # merge basic vehicle config into target vehicle config
        agent_configs = dict()
        num_agents = (
            ret_config["num_agents"] if ret_config["num_agents"] != -1 else SpawnManager.max_capacity(
                config["spawn_roads"],
                config["map_config"]["exit_length"],
                config["map_config"]["lane_num"],
            )
        )
        for id in range(num_agents):
            agent_id = "agent{}".format(id)
            config = copy.deepcopy(ret_config["vehicle_config"])
            if agent_id in ret_config["agent_configs"]:
                config["_specified_spawn_lane"] = (
                    True if "spawn_lane_index" in ret_config["agent_configs"][agent_id] else False
                )
                config["_specified_destination"] = (
                    True if "destination" in ret_config["agent_configs"][agent_id] else False
                )
                config.update(ret_config["agent_configs"][agent_id])
            agent_configs[agent_id] = config
            agent_configs[agent_id]["use_special_color"] = False
        ret_config["agent_configs"] = agent_configs
        # if ret_config["use_render"] and ret_config["disable_model_compression"]:
        #     logging.warning("Turn disable_model_compression=True can decrease the loading time!")

        if "prefer_track_agent" in config and config["prefer_track_agent"]:
            ret_config["agent_configs"][config["prefer_track_agent"]]["use_special_color"] = True
        ret_config["vehicle_config"]["random_agent_model"] = ret_config["random_agent_model"]
        return ret_config

    def setup_engine(self):
        super(MixedMultiAgentEnv, self).setup_engine()
        self.engine.update_manager("traffic_manager", MixedTrafficManager())

    def _find_in_range(self, v_id, distance):
        if distance <= 0:
            return [], []
        max_distance = distance
        dist_to_others = self.distance_map[v_id]
        dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
        ret = [
            dist_to_others_list[i] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        ret2 = [
            dist_to_others[dist_to_others_list[i]] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        return ret, ret2

    def _update_distance_map(self, dones=None):
        self.distance_map.clear()
        if hasattr(self, "vehicles_including_just_terminated"):
            vehicles = self.vehicles_including_just_terminated
            # if dones is not None:
            #     assert (set(dones.keys()) - set(["__all__"])) == set(vehicles.keys()), (dones, vehicles)
        else:
            vehicles = self.vehicles  # Fallback to old version MetaDrive, but this is not accurate!
        keys = [k for k, v in vehicles.items() if v is not None]
        for c1 in range(0, len(keys) - 1):
            for c2 in range(c1 + 1, len(keys)):
                k1 = keys[c1]
                k2 = keys[c2]
                p1 = vehicles[k1].position
                p2 = vehicles[k2].position
                distance = np.linalg.norm(p1 - p2)
                self.distance_map[k1][k2] = distance
                self.distance_map[k2][k1] = distance


class TSTFMixedMultiAgentEnv:
    """
    This class maintains a distance map of all agents and appends the
    neighbours' names and distances into info at each step.
    We should subclass this class to a base environment class.
    """

    @classmethod
    def default_config(cls):
        config = super(TSTFMixedMultiAgentEnv, cls).default_config()
        config["neighbours_distance"] = 50
        config["num_traffic_vehicle"] = 0.0
        return config

    def __init__(self, *args, **kwargs):
        super(TSTFMixedMultiAgentEnv, self).__init__(*args, **kwargs)
        self.bounding_box = None
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

    def step(self, actions):
        # o, r, d, i = super(MixedMultiAgentEnv, self).step(actions)
        o, r, tm, tc, i = super(TSTFMixedMultiAgentEnv, self).step(actions)
        for key in o:
            self.position_normalize(o[key][-2:])
        self._update_distance_map(dones=(tm or tc))
        global_reward = sum(r.values()) / len(r.values())
        for kkk in i.keys():
            i[kkk]["all_agents"] = list(i.keys())
            neighbours, nei_distances = self._find_in_range(kkk, self.config["neighbours_distance"])
            i[kkk]["neighbours"] = neighbours
            i[kkk]["neighbours_distance"] = nei_distances
            i[kkk]["global_rewards"] = global_reward

        return o, r, tm, tc, i

    def _post_process_config(self, config):
        from metadrive.manager.spawn_manager import SpawnManager
        config = super(MultiAgentMetaDrive, self)._post_process_config(config)
        ret_config = config
        # merge basic vehicle config into target vehicle config
        agent_configs = dict()
        num_agents = (
            ret_config["num_agents"] if ret_config["num_agents"] != -1 else SpawnManager.max_capacity(
                config["spawn_roads"],
                config["map_config"]["exit_length"],
                config["map_config"]["lane_num"],
            )
        )
        for id in range(num_agents):
            agent_id = "agent{}".format(id)
            config = copy.deepcopy(ret_config["vehicle_config"])
            if agent_id in ret_config["agent_configs"]:
                config["_specified_spawn_lane"] = (
                    True if "spawn_lane_index" in ret_config["agent_configs"][agent_id] else False
                )
                config["_specified_destination"] = (
                    True if "destination" in ret_config["agent_configs"][agent_id] else False
                )
                config.update(ret_config["agent_configs"][agent_id])
            agent_configs[agent_id] = config
            agent_configs[agent_id]["use_special_color"] = False
        ret_config["agent_configs"] = agent_configs
        # if ret_config["use_render"] and ret_config["disable_model_compression"]:
        #     logging.warning("Turn disable_model_compression=True can decrease the loading time!")

        if "prefer_track_agent" in config and config["prefer_track_agent"]:
            ret_config["agent_configs"][config["prefer_track_agent"]]["use_special_color"] = True
        ret_config["vehicle_config"]["random_agent_model"] = ret_config["random_agent_model"]
        return ret_config

    def setup_engine(self):
        super(TSTFMixedMultiAgentEnv, self).setup_engine()
        self.engine.update_manager("traffic_manager", MixedTrafficManager())

    def _find_in_range(self, v_id, distance):
        if distance <= 0:
            return [], []
        max_distance = distance
        dist_to_others = self.distance_map[v_id]
        dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
        ret = [
            dist_to_others_list[i] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        ret2 = [
            dist_to_others[dist_to_others_list[i]] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        return ret, ret2

    def _update_distance_map(self, dones=None):
        self.distance_map.clear()
        if hasattr(self, "vehicles_including_just_terminated"):
            vehicles = self.vehicles_including_just_terminated
            # if dones is not None:
            #     assert (set(dones.keys()) - set(["__all__"])) == set(vehicles.keys()), (dones, vehicles)
        else:
            vehicles = self.vehicles  # Fallback to old version MetaDrive, but this is not accurate!
        keys = [k for k, v in vehicles.items() if v is not None]
        for c1 in range(0, len(keys) - 1):
            for c2 in range(c1 + 1, len(keys)):
                k1 = keys[c1]
                k2 = keys[c2]
                p1 = vehicles[k1].position
                p2 = vehicles[k2].position
                distance = np.linalg.norm(p1 - p2)
                self.distance_map[k1][k2] = distance
                self.distance_map[k2][k1] = distance

    def _get_reset_return(self):
        self.bounding_box = self.current_map.road_network._get_bounding_box()
        self._update_distance_map()
        obses = super(TSTFMixedMultiAgentEnv, self)._get_reset_return()
        o = obses[0]
        for key in o:
            self.position_normalize(o[key][-2:])
        return obses

    def get_single_observation(self):
        original_obs = super(TSTFMixedMultiAgentEnv, self).get_single_observation()

        original_obs_cls = original_obs.__class__
        original_obs_name = original_obs_cls.__name__

        class PosObs(original_obs_cls):
            # @property
            # def observation_space(self):
            #     space = super(PosObs, self).observation_space
            #     # assert isinstance(space, Box)
            #     shape = list(space.shape)
            #     shape[0] += 2
            #     space = gymnasium.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)
            #     return space

            def observe(self, vehicle):
                ret = super(PosObs, self).observe(vehicle)
                position = vehicle.position
                ret = np.insert(ret, len(ret), position)
                return ret.astype(np.float32)

        PosObs.__name__ = original_obs_name
        PosObs.__qualname__ = original_obs_name

        return PosObs(self.config)

    def position_normalize(self, position):
        x_min, x_max, y_min, y_max = self.bounding_box
        position[0] = (position[0] - x_min) / (x_max - x_min)
        position[1] = (position[1] - y_min) / (y_max - y_min)
        return position


class LCFMixedMultiAgentEnv(MixedMultiAgentEnv):
    @classmethod
    def default_config(cls):
        config = super(LCFMixedMultiAgentEnv, cls).default_config()
        config.update(
            dict(
                # Overwrite the CCEnv's neighbours_distance=10 to 40.
                neighbours_distance=40,

                # Two mode to compute utility for each vehicle:
                # "linear": util = r_me * lcf + r_other * (1 - lcf), lcf in [0, 1]
                # "angle": util = r_me * cos(lcf) + r_other * sin(lcf), lcf in [0, pi/2]
                # "angle" seems to be more stable!
                lcf_mode="angle",
                lcf_dist="normal",  # "uniform" or "normal"
                lcf_normal_std=0.1,  # The initial STD of normal distribution, might change by calling functions.

                # If this is set to False, then the return reward is natively the LCF-weighted coordinated reward!
                # This will be helpful in ablation study!
                return_native_reward=True,

                # Whether to force set the lcf
                force_lcf=-100,
                enable_copo=True
            )
        )
        return config

    def __init__(self, config=None):
        super(LCFMixedMultiAgentEnv, self).__init__(config)
        self.lcf_map = {}
        assert hasattr(super(LCFMixedMultiAgentEnv, self), "_update_distance_map")
        assert self.config["lcf_mode"] in ["linear", "angle"]
        assert self.config["lcf_dist"] in ["uniform", "normal"]
        assert self.config["lcf_normal_std"] > 0.0
        self.force_lcf = self.config["force_lcf"]

        # Only used in normal LCF distribution
        # LCF is always in range [0, 1], but the real LCF degree is in [-pi/2, pi/2].
        self.current_lcf_mean = 0.0  # Set to 0 degree.
        self.current_lcf_std = self.config["lcf_normal_std"]

        self._last_obs = None
        self._traffic_light_counter = 0

    @property
    def enable_copo(self):
        return self.config["enable_copo"]

    def get_single_observation(self):
        original_obs = super(LCFMixedMultiAgentEnv, self).get_single_observation()

        if not self.enable_copo:
            return original_obs

        original_obs_cls = original_obs.__class__
        original_obs_name = original_obs_cls.__name__

        class LCFObs(original_obs_cls):
            @property
            def observation_space(self):
                space = super(LCFObs, self).observation_space
                # assert isinstance(space, Box)
                assert len(space.shape) == 1
                length = space.shape[0] + 1

                # Note that original metadrive obs space is [0, 1]
                space = gymnasium.spaces.Box(
                    low=np.array([-1.0] * length), high=np.array([1.0] * length), shape=(length,), dtype=space.dtype
                )
                space._shape = space.shape
                return space

        LCFObs.__name__ = original_obs_name
        LCFObs.__qualname__ = original_obs_name

        # TODO: This part is not beautiful! Refactor in future release!
        # from metadrive.envs.marl_envs.tinyinter import CommunicationObservation
        # if original_obs_cls == CommunicationObservation:
        #     return LCFObs(vehicle_config, self)
        # else:
        return LCFObs(self.config)

    def _get_reset_return(self):
        self.lcf_map.clear()
        self._update_distance_map()
        obses = super(LCFMixedMultiAgentEnv, self)._get_reset_return()

        ret = {}
        for k, o in obses[0].items():
            lcf, ret[k] = self._add_lcf(o)
            self.lcf_map[k] = lcf

        yet_another_new_obs = {}

        self._last_obs = ret
        return ret, obses[1]

    def step(self, actions):
        # step the environment
        # o, r, d, i = super(LCFEnv, self).step(actions)
        o, r, tm, tc, i = super(LCFMixedMultiAgentEnv, self).step(actions)
        assert set(i.keys()) == set(o.keys())
        new_obs = {}
        new_rewards = {}
        global_reward = sum(r.values()) / len(r.values())

        for agent_name, agent_info in i.items():
            assert "neighbours" in agent_info
            # Note: agent_info["neighbours"] records the neighbours within radius neighbours_distance.
            nei_rewards = [r[nei_name] for nei_name in agent_info["neighbours"]]
            if nei_rewards:
                i[agent_name]["nei_rewards"] = sum(nei_rewards) / len(nei_rewards)
            else:
                i[agent_name]["nei_rewards"] = 0.0  # Do not provide neighbour rewards if no neighbour
            i[agent_name]["global_rewards"] = global_reward

            # add LCF into observation, also update LCF map and info.
            agent_lcf, new_obs[agent_name] = self._add_lcf(
                agent_obs=o[agent_name], lcf=self.lcf_map[agent_name] if agent_name in self.lcf_map else None
            )
            if agent_name not in self.lcf_map:
                # The agent LCF is set for the whole episode
                self.lcf_map[agent_name] = agent_lcf
            i[agent_name]["lcf"] = agent_lcf
            i[agent_name]["lcf_deg"] = agent_lcf * 90

            # lcf_map stores values in [-1, 1]
            if self.config["lcf_mode"] == "linear":
                assert 0.0 <= agent_lcf <= 1.0
                new_r = agent_lcf * r[agent_name] + (1 - agent_lcf) * agent_info["nei_rewards"]
            elif self.config["lcf_mode"] == "angle":
                assert -1.0 <= agent_lcf <= 1.0
                lcf_rad = agent_lcf * np.pi / 2
                new_r = cos(lcf_rad) * r[agent_name] + sin(lcf_rad) * agent_info["nei_rewards"]
            else:
                raise ValueError("Unknown LCF mode: {}".format(self.config["lcf_mode"]))
            i[agent_name]["coordinated_rewards"] = new_r
            i[agent_name]["native_rewards"] = r[agent_name]
            if self.config["return_native_reward"]:
                new_rewards[agent_name] = r[agent_name]
            else:
                new_rewards[agent_name] = new_r

        yet_another_new_obs = {}
        self._last_obs = new_obs
        return new_obs, new_rewards, tm, tc, i

    def _add_lcf(self, agent_obs, lcf=None):

        if not self.enable_copo:
            return 0.0, agent_obs

        if self.force_lcf != -100:
            # Set LCF to given value
            if self.config["lcf_dist"] == "normal":
                assert -1.0 <= self.force_lcf <= 1.0
                lcf = get_np_random().normal(loc=self.force_lcf, scale=self.current_lcf_std)
                lcf = clip(lcf, -1, 1)
            else:
                lcf = self.force_lcf
        elif lcf is not None:
            pass
        else:
            # Sample LCF value from current distribution
            if self.config["lcf_dist"] == "normal":
                assert -1.0 <= self.current_lcf_mean <= 1.0
                lcf = get_np_random().normal(loc=self.current_lcf_mean, scale=self.current_lcf_std)
                lcf = clip(lcf, -1, 1)
            else:
                lcf = get_np_random().uniform(-1, 1)
        assert -1.0 <= lcf <= 1.0
        output_lcf = (lcf + 1) / 2  # scale to [0, 1]
        return lcf, np.float32(np.concatenate([agent_obs, [output_lcf]]))

    def set_lcf_dist(self, mean, std):
        assert self.enable_copo
        assert self.config["lcf_dist"] == "normal"
        self.current_lcf_mean = mean
        self.current_lcf_std = std
        assert std > 0.0
        assert -1.0 <= self.current_lcf_mean <= 1.0

    def set_force_lcf(self, v):
        assert self.enable_copo
        self.force_lcf = v


def get_mixed_multiagent_env(env_class):
    name = env_class.__name__

    class TMP(MixedMultiAgentEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


def get_tstf_mixed_multiagent_env(env_class):
    name = env_class.__name__

    class TMP(TSTFMixedMultiAgentEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


def get_lcf_mixed_multiagent_env(env_class):
    name = env_class.__name__

    class TMP(LCFMixedMultiAgentEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


def get_tstf_env(env_class, return_class=False):
    return get_rllib_compatible_env(get_tstf_mixed_multiagent_env(env_class), return_class)


def get_ippo_env(env_class, return_class=False):
    return get_rllib_compatible_env(get_mixed_multiagent_env(env_class), return_class)


def get_ccppo_env(env_class, return_class=False):
    return get_rllib_compatible_env(get_mixed_multiagent_env(env_class), return_class)


def get_copo_env(env_class, return_class=False):
    return get_rllib_compatible_env(get_lcf_mixed_multiagent_env(env_class), return_class)


def get_rllib_compatible_env(env_class, return_class=False):
    env_name = env_class.__name__
    env_class = createGymWrapper(env_class)
    env_class.__name__ = env_name

    class MA(env_class, MultiAgentEnv):
        _agent_ids = ["agent{}".format(i) for i in range(100)] + ["{}".format(i) for i in range(10000)] + ["sdc"]

        def __init__(self, config, *args, **kwargs):
            env_class.__init__(self, config, *args, **kwargs)
            MultiAgentEnv.__init__(self)

        @property
        def observation_space(self):
            ret = super(MA, self).observation_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            return ret

        @property
        def action_space(self):
            ret = super(MA, self).action_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            return ret

        def action_space_sample(self, agent_ids: list = None):
            """
            RLLib always has unnecessary stupid requirements that you have to bypass them by overwriting some
            useless functions.
            """
            return self.action_space.sample()

    # MA.__name__ = env_name
    MA.__qualname__ = env_name
    register_env(env_name, lambda config: MA(config))

    if return_class:
        return env_name, MA

    return env_name
