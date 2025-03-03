from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from ray import tune

from tstf.algo_copo.algo_copo import CoPOTrainer, USE_CENTRALIZED_CRITIC, USE_DISTRIBUTIONAL_LCF, COUNTERFACTUAL
from tstf.utils.callbacks import MultiAgentDrivingCallbacks
from tstf.utils.env_wrappers import get_copo_env
from tstf.utils.train import train
from tstf.utils.utils import get_train_parser

if __name__ == "__main__":
    exp_name = "COPO"

    # Setup config
    # We set the stop criterion to 2M environmental steps! Since PPO in single OurEnvironment converges at around 20k steps.
    stop = int(1200000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search(
            [
                get_copo_env(MultiAgentRoundaboutEnv),
            ]
        ),
        env_config=dict(num_agents=10,
                        traffic_mode="respawn",
                        num_traffic_vehicle=10,
                        map_config=dict(exit_length=60, lane_num=3),
                        vehicle_config=dict(use_special_color=True,),
                        # start_seed=tune.grid_search([4000, 6000, 8000]),
                        ),

        # ===== Resource =====
        # So we need 0.2(num_cpus_per_worker) * 5(num_workers) + 1(num_cpus_for_driver) = 2 CPUs per trial!
        # num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_gpus=0.5,
        num_cpus_per_worker=0.2,

        # ===== Meta SVO =====
        initial_svo_std=0.1,
        # **{USE_DISTRIBUTIONAL_SVO: tune.grid_search([True])},
        svo_lr=1e-4,
        svo_num_iters=5,
        use_global_value=True,
        num_workers=10,
        **{USE_CENTRALIZED_CRITIC: tune.grid_search([False])},
    )

    # Launch training
    train(
        CoPOTrainer,
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
