from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import m4_config as cfg


PACKAGE_FILES = (
    "__init__.py",
    "base.py",
    "observation_runtime.py",
    "kinematics_runtime.py",
    "combined_task.py",
    "landing_task.py",
    "m4_config.py",
    "m4_env.py",
    "randomizations.py",
    "spaces.py",
    "takeoff_task.py",
    "task_utils.py",
    "vehicle_adapters.py",
    "vehicle_profiles.py",
    "vehicle_specs.py",
)

STALE_FILES = (
    "m4tii_settings.py",
    "sitecustomize.py",
    "sync_to_isaaclab.py",
    "rl_games_ppo_config.yaml",
)


def _stamp_rl_games_name(path: Path, dry_run: bool) -> None:
    text = path.read_text()
    marker = "  config:\n"
    try:
        start = text.index(marker) + len(marker)
    except ValueError as exc:
        raise RuntimeError(f"Could not find RL-Games config block in {path}") from exc

    prefix, rest = text[:start], text[start:]
    stamped_rest, count = re.subn(
        r"(?m)^    name: .*$",
        f"    name: {cfg.RL_GAMES_EXPERIMENT_NAME}",
        rest,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not stamp params.config.name in {path}")
    stamped = prefix + stamped_rest
    if stamped != text and not dry_run:
        path.write_text(stamped)


def _copy_file(src: Path, dest: Path, dry_run: bool) -> None:
    if src.resolve() == dest.resolve():
        if dry_run:
            print(f"already in place {dest}")
        return
    if dry_run:
        print(f"would copy {src} -> {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def sync_task_package(dest_dir: Path, dry_run: bool) -> None:
    src_dir = Path(__file__).resolve().parent
    dest_dir = dest_dir.expanduser().resolve()
    if not dest_dir.exists():
        raise FileNotFoundError(f"Destination does not exist: {dest_dir}")

    for filename in PACKAGE_FILES:
        src = src_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        _copy_file(src, dest_dir / filename, dry_run)

    src_yaml = _find_rl_games_yaml(src_dir)

    for filename in STALE_FILES:
        stale = dest_dir / filename
        if stale.exists():
            if dry_run:
                print(f"would remove stale {stale}")
            else:
                stale.unlink()

    agents_yaml = dest_dir / "agents" / "rl_games_ppo_cfg.yaml"
    if not agents_yaml.parent.exists():
        raise FileNotFoundError(f"Expected RL-Games config directory does not exist: {agents_yaml.parent}")
    _copy_file(src_yaml, agents_yaml, dry_run)
    if dry_run:
        print(f"would stamp {agents_yaml} name -> {cfg.RL_GAMES_EXPERIMENT_NAME}")
    else:
        _stamp_rl_games_name(agents_yaml, dry_run=False)

    if not dry_run:
        base_source = (dest_dir / "base.py").read_text()
        combined_source = (dest_dir / "combined_task.py").read_text()
        base_markers = (
            "episode_log_keys: tuple[str, ...] = ()",
            "self._logged_episode_reward_keys = tuple(dict.fromkeys((*reward_keys, *episode_log_keys)))",
        )
        combined_markers = (
            'super().__init__(\n            env,\n            cfg,\n            self.reward_keys,',
            '"takeoff_timing_multiplier": takeoff_event.float()',
        )
        missing_base_markers = [marker for marker in base_markers if marker not in base_source]
        missing_combined_markers = [marker for marker in combined_markers if marker not in combined_source]
        if missing_base_markers or missing_combined_markers:
            raise RuntimeError(
                "Synced task package is missing takeoff timing logging support: "
                + ", ".join(missing_base_markers + missing_combined_markers)
            )
        print("Verified base.py and combined_task.py include takeoff_timing_multiplier logging")

    print(f"{'Dry run sync complete' if dry_run else 'Synced'}: {dest_dir}")
    print(f"RL-Games experiment name: {cfg.RL_GAMES_EXPERIMENT_NAME}")


def _find_rl_games_yaml(src_dir: Path) -> Path:
    candidates = (
        src_dir / "rl_games_ppo_config.yaml",
        src_dir / "agents" / "rl_games_ppo_cfg.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find RL-Games YAML. Expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _checkpoint_args() -> list[str]:
    checkpoint = str(getattr(cfg, "CHECKPOINT", "") or "").strip()
    if not checkpoint:
        return []
    return ["--checkpoint", checkpoint]


def build_command(mode: str) -> list[str]:
    mode = mode.lower()
    task_arg = f"--task={cfg.ISAACLAB_TASK_ID}"
    if mode == "train":
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={int(cfg.TRAIN_NPROC_PER_NODE)}",
            "source/standalone/workflows/rl_games/train.py",
            task_arg,
        ]
        if cfg.TRAIN_HEADLESS:
            cmd.append("--headless")
        if cfg.TRAIN_DISTRIBUTED:
            cmd.append("--distributed")
        cmd.extend(["--num_envs", str(int(cfg.TRAIN_NUM_ENVS))])
        cmd.extend(_checkpoint_args())
        return cmd

    if mode == "play":
        cmd = [
            sys.executable,
            "source/standalone/workflows/rl_games/play.py",
            task_arg,
            "--num_envs",
            str(int(cfg.PLAY_NUM_ENVS)),
        ]
        if cfg.PLAY_HEADLESS:
            cmd.append("--headless")
        cmd.extend(_checkpoint_args())
        return cmd

    raise ValueError(f"Unknown MODE '{mode}'. Expected 'train' or 'play'.")


def _format_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def _prepend_pythonpath(env: dict[str, str], paths: tuple[Path, ...]) -> None:
    entries = [str(path) for path in paths]
    existing = env.get("PYTHONPATH", "")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the M4 task config and run the configured train/play command.")
    parser.add_argument("--dry-run", action="store_true", help="Print sync actions and command without executing.")
    parser.add_argument("--sync-only", action="store_true", help="Sync files and stamp YAML, but do not launch.")
    parser.add_argument("--no-sync", action="store_true", help="Launch without syncing files first.")
    parser.add_argument("--mode", choices=("train", "play"), help="Override m4_config.MODE for this launch.")
    args = parser.parse_args()

    mode = args.mode or cfg.MODE
    dest_dir = Path(cfg.ISAACLAB_TASK_DIR)
    isaaclab_root = Path(cfg.ISAACLAB_ROOT)

    if not args.no_sync:
        sync_task_package(dest_dir, dry_run=args.dry_run)

    cmd = build_command(mode)
    print("Command:")
    print(_format_command(cmd))

    if args.dry_run or args.sync_only:
        return

    env = os.environ.copy()
    env["M4_RUN_MODE"] = mode.lower()
    _prepend_pythonpath(env, (Path(__file__).resolve().parent, dest_dir))
    env.setdefault("M4_TB_SCALAR_DEBUG", "1")
    env.setdefault("M4_NORMAL_SCALE_DEBUG", "0")
    subprocess.run(cmd, cwd=isaaclab_root, check=True, env=env)


if __name__ == "__main__":
    main()
