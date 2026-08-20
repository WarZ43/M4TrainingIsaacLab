from __future__ import annotations

# Main run switchboard.
ROBOT = "atmo"
TASK = "combined"
STAGE = 1
# Launch switchboard.
MODE = "train"  # "train" or "play"
CHECKPOINT = ""
# "/home/warren/IsaacLab/logs/rl_games/atmo_combined_stage1/2026-07-29_13-33-26/nn/last_atmo_combined_stage1_ep_2050_rew_2216.1643.pth"
# "/home/warren/IsaacLab/logs/rl_games/m4tii_combined_stage3/2026-07-24_12-47-35/nn/last_m4tii_combined_stage3_ep_8300_rew_1509.7721.pth"
# "/home/warren/IsaacLab/logs/rl_games/m4tii_combined_stage1/2026-07-21_00-00-05/nn/last_m4tii_combined_stage1_ep_900_rew_1153.37.pth"
# Set this to the checkpoint/trainer epoch when resuming so curriculum ramps use absolute epochs.
RAMP_EPOCH_OFFSET = 0
# Set this to the epoch where the selected stage first began. Keep it unchanged
# when resuming within that stage so action-authority ramps do not restart.
STAGE_RAMP_START_EPOCH = 0.0

# IsaacLab task registration and paths.
ISAACLAB_ROOT = "/home/warren/IsaacLab"
ISAACLAB_TASK_ID = "m4"
ISAACLAB_TASK_DIR = "/home/warren/IsaacLab/source/extensions/omni.isaac.lab_tasks/" "omni/isaac/lab_tasks/direct/atmo"

# Train command settings.
TRAIN_NUM_ENVS = 16384
TRAIN_NPROC_PER_NODE = 2    
TRAIN_HEADLESS = True
TRAIN_DISTRIBUTED = True

# Play command settings.
PLAY_NUM_ENVS = 64
PLAY_HEADLESS = False


def experiment_name(robot: str = ROBOT, task: str = TASK, stage: int = STAGE) -> str:
    return f"{robot}_{task}_stage{int(stage)}"


RL_GAMES_EXPERIMENT_NAME = experiment_name()
