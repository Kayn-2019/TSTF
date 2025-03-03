import argparse
import os
import cv2  # 新增导入cv2库
import numpy as np
from datetime import datetime
from pathlib import Path
import math

from tstf.utils.env_wrappers import get_copo_env
from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv
from ray.rllib.policy.policy import Policy

# 创建视频保存目录
VIDEO_DIR = Path("./simulation_videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def save_frames_to_video(frames, episode_num, success_rate, resolution=(1000, 1000)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = VIDEO_DIR / "different_map" / "4_lane_intersection" /f"copo_4_lane_intersection_success_{success_rate:.2f}.mp4"

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(filename), fourcc, 30, resolution)

    # 转换并写入帧
    for frame in frames:
        # 转换颜色空间 RGB -> BGR
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    video_writer.release()
    print(f"Saved simulation video to: {filename}")


def save_frames_to_avi(frames, episode_num, success_rate, resolution=(1000, 1000)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = VIDEO_DIR / "different_map" / "4_lane_intersection" /f"copo_4_lane_intersection_success_{success_rate:.2f}.avi"  # 扩展名改为.avi

    # 关键修改：使用AVI兼容的编码器（如XVID、DIVX）
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或 'DIVX', 'MJPG'
    video_writer = cv2.VideoWriter(str(filename), fourcc, 30, resolution)

    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    video_writer.release()
    print(f"Saved AVI video to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",
                        default="../CoPOTrainer_MultiAgentRoundaboutEnv_96346_00000_0_env=MultiAgentRoundaboutEnv,start_seed=4000,seed=0,use_centralized_critic=False_2025-02-21_10-36-00/checkpoint_000600",
                        type=str)
    args = parser.parse_args()

    # ===== 加载训练好的策略 =====
    checkpoint_path = args.checkpoint_path
    policy = Policy.from_checkpoint(checkpoint_path)["default"]

    # ===== 创建环境 =====
    env_name, env_class = get_copo_env(MultiAgentIntersectionEnv, return_class=True)
    env_config = dict(
        num_agents=10,
        traffic_mode="respawn",
        num_traffic_vehicle=10,
        map_config=dict(exit_length=60, lane_num=4),
        vehicle_config=dict(
            use_special_color=True,
        ),
        # use_render=True  # 确保启用渲染
    )
    env = env_class(env_config)

    # ===== 主循环 =====
    for episode_num in range(1000):  # 控制总episode数量
        o = env.reset()
        d = {"__all__": False}
        ep_success = 0
        ep_step = 0
        ep_agent = 0
        info = None
        # frames = []
        while not d["__all__"]:
            obs_to_be_eval = []
            obs_to_be_eval_keys = []
            for agent_id, agent_ob in o.items():  # I don't know why there is one 'agent0' extra here!
                if (agent_id not in d) or (not d.get(agent_id, False)):
                    obs_to_be_eval.append(agent_ob)
                    obs_to_be_eval_keys.append(agent_id)
            input_dict = {"obs": obs_to_be_eval}
            actions, _, _ = policy.compute_actions_from_input_dict(input_dict)
            action_to_send = {}
            for count, agent_id in enumerate(obs_to_be_eval_keys):
                action_to_send[agent_id] = actions[count]

            o, r, d, info = env.step(action_to_send)
            ep_step += 1

            # 捕获当前帧
            env.render(
                mode="top_down",
                window=True,
                screen_record=True,
                screen_size=(1000, 1000),
                # camera_position=(120, -10),
                scaling=5.5,
                film_size=(1000, 1000),
            )
            # frames.append(frame)

            # 更新统计信息
            for agent_id in d:
                if agent_id != "__all__" and d[agent_id]:
                    ep_agent += 1
                    if info[agent_id].get("arrive_dest", False):
                        ep_success += 1

        # Episode结束处理
        success_rate = ep_success / ep_agent if ep_agent > 0 else 0
        print(f"Episode {episode_num} completed. Success rate: {success_rate:.2%}")

        # 保存条件判断
        if 0.71 <= success_rate <= 0.73:
            frames = env.top_down_renderer.screen_frames
            save_frames_to_avi(frames, episode_num, success_rate)
            save_frames_to_video(frames, episode_num, success_rate)

        # 重置环境
        env.close()

# 注意：确保在运行前安装依赖
# pip install opencv-python numpy ray[rllib] metadrive
