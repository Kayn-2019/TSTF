from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from ray import tune
from tstf.algo_tstf.tstf import TSTF
from tstf.utils.callbacks import MultiAgentDrivingCallbacks
from tstf.utils.train import train
from tstf.utils.env_wrappers import get_tstf_env

if __name__ == "__main__":
    exp_name = "TSTF_NORMAL"

    # Setup config
    stop = int(1200000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search(
            [
                get_tstf_env(MultiAgentRoundaboutEnv),
            ]
        ),
        env_config=dict(
            num_agents=10,
            traffic_mode="respawn",
            num_traffic_vehicle=10,
            map_config=dict(exit_length=60, lane_num=3),
            vehicle_config=dict(use_special_color=True,),
            # start_seed=tune.grid_search([4000, 5000, 6000, 7000, 8000, 9000]),
            # use_render=True,
        ),
        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        # lr_schedule=tune.grid_search([[(0, 1e-3), (600000, 1e-5)],
        #                               [(0, 1e-3), (500000, 1e-5)],
        #                               [(0, 1e-3), (400000, 1e-5)],]),
        num_gpus=0.5,  #for local worker
        num_workers=10,  # besides localworker, num of rollout env samplers
        num_cpus_for_local_worker=5,
        num_cpus_per_worker=0.5,
    )

    # Launch training
    train(
        TSTF,
        exp_name=exp_name,
        keep_checkpoints_num=10,
        stop=stop,
        config=config,
        num_gpus=1,
        num_seeds=1,
        custom_callback=MultiAgentDrivingCallbacks,
        checkpoint_freq=50,
        # local_mode=True,
    )
