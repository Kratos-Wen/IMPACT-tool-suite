#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a Python script in a selected conda environment profile.

Usage:
  python tools/runners/run_in_env.py --profile asot -- tools/asot_full_infer_adapter.py --help
  python tools/runners/run_in_env.py --profile mobileclip -- tools/mobileclip_infer.py --video xxx.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List


PROFILE_DEFAULT_ENVS: Dict[str, str] = {
    "ui": "current",
    "opentad": "current",
    "east": "current",
    "mobileclip": "mobileclip",
    "asot": "current",
}


def _default_env_config_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw = str(os.environ.get("RUNNER_ENV_CONFIG", "runner_envs.json") or "").strip()
    if not raw:
        raw = "runner_envs.json"
    if not os.path.isabs(raw):
        raw = os.path.abspath(os.path.join(repo_root, raw))
    return raw


def _load_env_config() -> Dict[str, str]:
    cfg_path = _default_env_config_path()
    if not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("profiles"), dict):
        payload = payload.get("profiles")
    if not isinstance(payload, dict):
        return {}
    resolved: Dict[str, str] = {}
    for k, v in payload.items():
        key = str(k or "").strip().lower()
        val = str(v or "").strip()
        if key and val:
            resolved[key] = val
    return resolved


def _resolve_env_name(profile: str, explicit_env: str) -> str:
    if explicit_env:
        return str(explicit_env).strip()
    key = str(profile or "opentad").strip().lower() or "opentad"

    env_key = f"RUNNER_ENV_{key.upper()}"
    if os.environ.get(env_key):
        return str(os.environ.get(env_key) or "").strip()

    if key == "asot" and os.environ.get("ASOT_CONDA_ENV"):
        return str(os.environ.get("ASOT_CONDA_ENV") or "").strip()
    if key == "asot" and os.environ.get("ASOT_CONDA_PREFIX"):
        return str(os.environ.get("ASOT_CONDA_PREFIX") or "").strip()
    if key == "mobileclip" and os.environ.get("MOBILECLIP_CONDA_ENV"):
        return str(os.environ.get("MOBILECLIP_CONDA_ENV") or "").strip()
    if key == "mobileclip" and os.environ.get("MOBILECLIP_CONDA_PREFIX"):
        return str(os.environ.get("MOBILECLIP_CONDA_PREFIX") or "").strip()
    if key in {"opentad", "east", "ui"} and os.environ.get("OPENTAD_CONDA_ENV"):
        return str(os.environ.get("OPENTAD_CONDA_ENV") or "").strip()
    if key in {"opentad", "east", "ui"} and os.environ.get("OPENTAD_CONDA_PREFIX"):
        return str(os.environ.get("OPENTAD_CONDA_PREFIX") or "").strip()

    config_map = _load_env_config()
    if config_map.get(key):
        return str(config_map.get(key) or "").strip()

    # Common local layout fallback: /home/.../conda_envs/<name>
    home = os.path.expanduser("~")
    if key in {"asot", "mobileclip"}:
        cand = os.path.join(home, "IsaacDrive", "conda_envs", key)
        if os.path.isdir(cand):
            return cand

    return PROFILE_DEFAULT_ENVS.get(key, key)


def _build_exec_cmd(
    script_path: str,
    script_args: List[str],
    profile: str,
    env_name: str,
    conda_exe: str,
    python_bin: str,
) -> List[str]:
    target_env = _resolve_env_name(profile=profile, explicit_env=env_name)
    script_abs = os.path.abspath(os.path.expanduser(script_path))

    # Allow opting out of conda by setting env to current/self/none.
    if target_env.lower() in {"", "current", "self", "none"}:
        return [sys.executable, script_abs, *script_args]

    if os.path.isdir(target_env):
        return [conda_exe, "run", "--no-capture-output", "-p", target_env, python_bin, script_abs, *script_args]
    return [conda_exe, "run", "--no-capture-output", "-n", target_env, python_bin, script_abs, *script_args]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a script in a configured conda environment profile.")
    ap.add_argument("--profile", default="opentad", help="Environment profile: opentad/east/mobileclip/asot/...")
    ap.add_argument("--env-name", default="", help="Override conda env name directly.")
    ap.add_argument("--conda-exe", default=os.environ.get("CONDA_EXE", "conda"), help="conda executable path.")
    ap.add_argument("--python-bin", default="python", help="Python binary name inside target env.")
    ap.add_argument("--cwd", default="", help="Optional working directory.")
    ap.add_argument("--print-cmd", action="store_true", help="Print resolved command before running.")
    args, rest = ap.parse_known_args()

    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("[RUNNER][ERROR] Missing script path. Use: run_in_env.py --profile <p> -- <script.py> <args...>", file=sys.stderr)
        return 2

    script_path = rest[0]
    script_args = rest[1:]
    cmd = _build_exec_cmd(
        script_path=script_path,
        script_args=script_args,
        profile=args.profile,
        env_name=args.env_name,
        conda_exe=args.conda_exe,
        python_bin=args.python_bin,
    )

    if args.print_cmd:
        print("[RUNNER] " + " ".join(cmd))

    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        proc = subprocess.Popen(cmd, cwd=(args.cwd or None), env=child_env)
    except FileNotFoundError as exc:
        print(f"[RUNNER][ERROR] Command not found: {exc}", file=sys.stderr)
        return 127
    except Exception as exc:
        print(f"[RUNNER][ERROR] Failed to launch process: {exc}", file=sys.stderr)
        return 1
    return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
