
from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from ray import tune
from tstf.utils.callbacks import MultiAgentDrivingCallbacks
from tstf.utils.train import train
from tstf.utils.utils import get_train_parser
from tstf.utils.env_wrappers import get_ippo_env
from tstf.algo_ippo.algo_ippo import IPPOTrainer


if __name__ == "__main__":
    exp_name = "IPPO_NORMAL"

    # Setup config
    stop = int(1200000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search(
            [
                get_ippo_env(MultiAgentRoundaboutEnv),
            ]
        ),
        env_config=dict(num_agents=10,
                        traffic_mode="respawn",
                        num_traffic_vehicle=10,
                        map_config=dict(exit_length=60, lane_num=3),
                        vehicle_config=dict(use_special_color=True,),
                        # start_seed=tune.grid_search([4000, 6000, 8000]),
                        # use_render=True,
                        ),
        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=0.25,
        num_cpus_per_worker=0.2,
        num_worker=10,
    )

    # Launch training
    train(
        IPPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=10,
        stop=stop,
        config=config,
        num_gpus=1,
        num_seeds=1,
        custom_callback=MultiAgentDrivingCallbacks,
        checkpoint_freq=50,
        # local_mode=True
    )
