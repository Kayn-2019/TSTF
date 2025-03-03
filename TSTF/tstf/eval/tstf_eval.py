import argparse
import os


from tstf.utils.env_wrappers import get_tstf_env
from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv
from ray.rllib.policy.policy import Policy
import numpy as np



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",
                        default="../TSTF_MultiAgentRoundaboutEnv_ff685_00002_2_env=MultiAgentRoundaboutEnv,start_seed=6000,seed=0_2025-02-21_03-17-09/checkpoint_000600",
                        type=str,)
    args = parser.parse_args()

    # ===== Load trained policy =====
    checkpoint_path = args.checkpoint_path

    policy = Policy.from_checkpoint(checkpoint_path)["default"]
    # algorithm = Algorithm.from_checkpoint(checkpoint_path)
    # ===== Create environment =====
    env_name,env_class = get_tstf_env(MultiAgentRoundaboutEnv, return_class=True)
    env_config = dict(num_agents=10,
                      traffic_mode="respawn",
                      num_traffic_vehicle=10,
                      # use_render=True,
                      map_config=dict(exit_length=60, lane_num=4),
                      # horizon=1000,
                      vehicle_config=dict(use_special_color=True,
                                          # lidar=dict(
                                          # num_others=4, distance=30)
                                          )
                      )
    env = env_class(env_config)

    # ===== Running =====
    o = env.reset()
    d = {"__all__": False}
    ep_success = 0
    ep_step = 0
    ep_agent = 0
    info = None
    for i in range(100000000000000):
        processed_obs = []
        obs_to_be_eval_keys = []
        num_agents = 5
        for agent_id, obs in o.items():
            if (agent_id not in d) or (not d.get(agent_id, False)):
                region_obs = []
                region_obs.append(obs)
                if info is not None and (len(info) != 0):
                    agent_info = info[agent_id]
                    for i, nei_name in enumerate(agent_info["neighbours"]):
                        if i >= num_agents - 1:
                            break
                        region_obs.append(o[nei_name])
                if len(region_obs) < num_agents:
                    num_pad = num_agents - len(region_obs)
                    for _ in range(num_pad):
                        region_obs.append(np.zeros(len(obs), dtype='float32'))
                region_obs_np = np.concatenate(region_obs)
                processed_obs.append(region_obs_np)
                obs_to_be_eval_keys.append(agent_id)
        # obs_to_be_eval = []
        # obs_to_be_eval_keys = []
        # for agent_id, agent_ob in o.items():  # I don't know why there is one 'agent0' extra here!
        #     if (agent_id not in d) or (not d.get(agent_id, False)):
        #         obs_to_be_eval.append(agent_ob)
        #         obs_to_be_eval_keys.append(agent_id)
        input_dict = {"obs":processed_obs}
        actions,_,_ = policy.compute_actions_from_input_dict(input_dict)
        action_to_send = {}
        for count, agent_id in enumerate(obs_to_be_eval_keys):
            action_to_send[agent_id] = actions[count]

        o, r, d, info = env.step(action_to_send)
        ep_step += 1


        for kkk, ddd in d.items():
            if kkk != "__all__" and ddd:
                ep_success += 1 if info[kkk]["arrive_dest"] else 0
                ep_agent += 1

        # env.render(mode="topdown",
        #            window=True,
        #            screen_record=True,
        #            screen_size=(1500, 1500),
        #            camera_position=(120, -10),
        #            text={
        #                "total agents": ep_agent,
        #                "existing agents": len(o),
        #                "success rate": ep_success / ep_agent if ep_agent > 0 else None,
        #                "ep step": ep_step
        #            }
        #
        #            )
        if d["__all__"]:  # This is important!
            if ep_success / ep_agent >= 0.65:
                print(d, info)
                print("Episode success rate: ", ep_success / ep_agent if ep_agent > 0 else None)
                print(
                    {
                        "total agents": ep_agent,
                        "existing agents": len(o),
                        "success rate": ep_success / ep_agent if ep_agent > 0 else None,
                        "ep step": ep_step
                    }
                )
            # env.top_down_renderer.generate_gif(f"tftt_roundabout_20_4lanes_{i}.gif")
            ep_success = 0
            ep_step = 0
            ep_agent = 0


            o = env.reset()
            d = {"__all__": False}
            info = None

            # policy_function.reset()
            # break

        # if True:
        #     env.render(
        #         text={
        #             "total agents": ep_agent,
        #             "existing agents": len(o),
        #             "success rate": ep_success / ep_agent if ep_agent > 0 else None,
        #             "ep step": ep_step
        #         }
        #     )
        # else:
        #     env.render(mode="top_down", num_stack=25)

    env.close()
