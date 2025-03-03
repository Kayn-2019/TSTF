from metadrive.envs.marl_envs import MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, \
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentIntersectionEnv
from ray import tune

from tstf.algo_ccppo.algo_ccppo import CCPPOTrainer
from tstf.utils.callbacks import MultiAgentDrivingCallbacks
from tstf.utils.train import train
from tstf.utils.env_wrappers import get_ccppo_env

if __name__ == "__main__":
    exp_name = "CCPO_MF"
    stop = int(120_0000)
    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search(
            [
                get_ccppo_env(MultiAgentRoundaboutEnv),
            ]
        ),
        env_config=dict(num_agents=10,
                        traffic_mode="respawn",
                        num_traffic_vehicle=10,
                        map_config=dict(exit_length=60, lane_num=3),
                        vehicle_config=dict(use_special_color=True,),
                        # start_seed=tune.grid_search([4000,  6000, 7000,]),
                        # use_render=True,
                        ),

        # ===== Resource =====
        num_gpus=0.25,

        # ===== MAPPO =====
        counterfactual=True,
        use_mode=tune.grid_search(["mf", "concat"]),
        # use_mode = "mf",
        mf_nei_distance=10,
    )

    # Launch training
    train(
        CCPPOTrainer,
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
