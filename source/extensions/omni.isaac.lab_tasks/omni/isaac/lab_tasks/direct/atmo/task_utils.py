from __future__ import annotations

import torch


def heading_yaw_from_quat(quat_wxyz: torch.Tensor, forward_yaw_offset: float) -> torch.Tensor:
    """Yaw of the vehicle's rolling-forward axis, in world frame."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw + float(forward_yaw_offset)


def to_heading_frame(vector_w: torch.Tensor, heading_yaw: torch.Tensor) -> torch.Tensor:
    """Rotate a world vector about z only, into the vehicle's heading frame.

    Yaw alone, deliberately: gravity is world vertical, so thrust, altitude and
    climb rate are all gravity referenced. Folding roll and pitch into the
    transform would mix the vertical channel into the horizontal ones and leave
    the policy to undo it. Yaw is also the only rotation the task is symmetric
    under, since the trajectory does not care which compass heading it runs
    along but does care about up. The z component is therefore passed through.

    +x is the rolling-forward direction, so the sign of the x component answers
    whether driving forward closes an error or opens it.
    """
    cos_yaw = torch.cos(heading_yaw)
    sin_yaw = torch.sin(heading_yaw)
    return torch.stack(
        (
            cos_yaw * vector_w[:, 0] + sin_yaw * vector_w[:, 1],
            -sin_yaw * vector_w[:, 0] + cos_yaw * vector_w[:, 1],
            vector_w[:, 2],
        ),
        dim=1,
    )


def rotation_matrix_to_heading_frame(
    matrix_w: torch.Tensor, heading_yaw: torch.Tensor
) -> torch.Tensor:
    """Strip yaw from a body-to-world rotation, leaving tilt relative to heading.

    Absolute yaw is uniformly random at reset and carries no task information,
    so leaving it in makes the attitude observation mostly nuisance. The heading
    the task does care about is already carried by the yaw error term.
    """
    cos_yaw = torch.cos(heading_yaw).unsqueeze(1)
    sin_yaw = torch.sin(heading_yaw).unsqueeze(1)
    row_x = matrix_w[:, 0, :]
    row_y = matrix_w[:, 1, :]
    return torch.stack(
        (
            cos_yaw * row_x + sin_yaw * row_y,
            -sin_yaw * row_x + cos_yaw * row_y,
            matrix_w[:, 2, :],
        ),
        dim=1,
    )


class RewardMixer:
    def __init__(self, reward_keys: tuple[str, ...]):
        self.reward_keys = reward_keys

    def sum(self, rewards: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.stack([rewards[key] for key in self.reward_keys]), dim=0)


def stage_value(env, values, name: str):
    try:
        if hasattr(env, "curriculum_stage_for_value"):
            stage = int(env.curriculum_stage_for_value(name))
        else:
            stage = int(env.cfg.curriculum_stage)
        if stage < 1:
            raise ValueError(f"curriculum_stage must be >= 1, got {stage}")
        stage_idx = stage - 1
        try:
            return values[stage_idx]
        except IndexError as exc:
            if getattr(env.cfg, "reuse_last_stage_value", True):
                return values[-1]
            raise ValueError(f"curriculum_stage={stage} has no entry for {name}") from exc
    except TypeError:
        return values
