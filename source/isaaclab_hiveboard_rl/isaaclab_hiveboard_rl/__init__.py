import gymnasium as gym

import isaaclab_hiveboard

from isaaclab_hiveboard_rl import env, specs

for robot, tool in specs.PAIRS:
    for play in (False, True):
        gym.register(
            id=specs.task_id(robot, tool, play),
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": lambda r=robot, t=tool, p=play: env.make_env_cfg(r, t, p),
                "rsl_rl_cfg_entry_point": lambda r=robot, t=tool: (
                    env.AnymalBallValveStudentPPORunnerCfg(experiment_name=f"{r}_{t}_student_ppo")
                ),
                "default_agent": "rsl_rl",
            },
        )
