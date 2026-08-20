# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
M4 landing environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="m4",
    entry_point=f"{__name__}.m4_env:M4Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.m4_env:M4EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M4PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# The Ioannis-lineage baseline runs the self-contained atmo_env.py, which shares
# nothing with the M4 framework beyond the asset. It needs its own id: "m4" is
# wired to M4Env and stamping m4_config cannot reach this env at all.
gym.register(
    id="atmo_baseline",
    entry_point=f"{__name__}.atmo_env:ATMOEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atmo_env:ATMOEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
