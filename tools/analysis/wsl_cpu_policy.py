#!/usr/bin/env python3
"""Verify the local WSL CPU split: 24 online, 20 default, 24 serving."""
from __future__ import annotations

import argparse
import configparser
import json
import pathlib
import subprocess
from typing import Any


def parse_cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("CPU set is empty")
    return cpus


def parse_wsl_processors(text: str) -> int:
    parser = configparser.ConfigParser()
    parser.read_string(text)
    return parser.getint("wsl2", "processors")


def evaluate_policy(snapshot: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "configured_processors": 24,
        "kernel_online": set(range(24)),
        "experiment_allowed": set(range(20)),
        "serving_allowed": set(range(24)),
    }
    observed = {
        "configured_processors": int(snapshot["configured_processors"]),
        "kernel_online": parse_cpu_set(str(snapshot["kernel_online"])),
        "experiment_allowed": parse_cpu_set(str(snapshot["experiment_allowed"])),
        "serving_allowed": parse_cpu_set(str(snapshot["serving_allowed"])),
    }
    errors = []
    for field, expected_value in expected.items():
        if observed[field] != expected_value:
            errors.append(
                f"{field}: expected {len(expected_value) if isinstance(expected_value, set) else expected_value}, "
                f"observed {len(observed[field]) if isinstance(observed[field], set) else observed[field]}"
            )
    return {
        "ok": not errors,
        "configured_processors": observed["configured_processors"],
        "kernel_online_vcpus": len(observed["kernel_online"]),
        "experiment_vcpus": len(observed["experiment_allowed"]),
        "serving_vcpus": len(observed["serving_allowed"]),
        "service_main_pid": snapshot.get("service_main_pid"),
        "errors": errors,
    }


def _wsl_snapshot(distro: str) -> dict[str, str]:
    def run(*command: str) -> str:
        return subprocess.run(
            ["wsl.exe", "-d", distro, "--", *command],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def allowed_from_status(status: str, label: str) -> str:
        allowed_line = next(
            (line for line in status.splitlines() if line.startswith("Cpus_allowed_list:")),
            None,
        )
        if allowed_line is None:
            raise RuntimeError(f"{label} process status has no Cpus_allowed_list")
        return allowed_line.split(":", 1)[1].strip()

    service_pid = run(
        "systemctl", "show", "llm-inference.service", "-p", "MainPID", "--value"
    )
    if not service_pid.isdecimal() or int(service_pid) <= 0:
        raise RuntimeError("llm-inference.service has no live MainPID")
    return {
        "kernel_online": run("cat", "/sys/devices/system/cpu/online"),
        "experiment_allowed": allowed_from_status(
            run("cat", "/proc/self/status"), "experiment"
        ),
        "service_main_pid": service_pid,
        "serving_allowed": allowed_from_status(
            run("cat", f"/proc/{service_pid}/status"), "service"
        ),
    }


def collect_policy(distro: str, wslconfig: pathlib.Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = _wsl_snapshot(distro)
    snapshot["configured_processors"] = parse_wsl_processors(
        wslconfig.read_text(encoding="utf-8-sig")
    )
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", default="Ubuntu-24.04")
    parser.add_argument(
        "--wslconfig",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".wslconfig",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate_policy(collect_policy(args.distro, args.wslconfig))
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        report = {"ok": False, "errors": [detail]}
    except Exception as error:
        report = {"ok": False, "errors": [str(error)]}
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["ok"]:
        print(
            "WSL CPU POLICY: PASS "
            f"kernel={report['kernel_online_vcpus']} "
            f"experiments={report['experiment_vcpus']} "
            f"serving={report['serving_vcpus']}"
        )
    else:
        print("WSL CPU POLICY: FAIL")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
