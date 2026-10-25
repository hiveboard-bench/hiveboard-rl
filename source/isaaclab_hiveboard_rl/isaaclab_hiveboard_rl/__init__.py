import os

import gymnasium as gym

from isaaclab_hiveboard_rl import env


def camel(name):
    return "".join(word.capitalize() for word in name.split("_"))


for robot in sorted(os.listdir(os.path.join(env.REPO_ROOT, "configs", "robots"))):
    for task in sorted(os.listdir(os.path.join(env.REPO_ROOT, "configs", "tasks"))):
        robot, task = robot.removesuffix(".yaml"), task.removesuffix(".yaml")
        for mode, suffix in (("train", ""), ("play", "-Play")):
            gym.register(
                id=f"Isaac-HiveBoard-{camel(robot)}-{camel(task)}-RL{suffix}-v0",
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": lambda r=robot, t=task, m=mode: env.compose(r, t, m),
                    "rsl_rl_cfg_entry_point": lambda r=robot, t=task: env.runner_cfg(r, t),
                    "default_agent": "rsl_rl",
                },
            )
