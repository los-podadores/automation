"""Cluster management CLI for Los Podadores Ray RLlib cluster.

Commands:
  up     - Deploy worker scripts and maintain parallel SSH worker connections
  down   - Stop Ray on all remote workers
  status - Query Ray cluster health and display active nodes
  test   - Run distributed tasks across all connected nodes
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Final

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cluster_config import (  # noqa: E402
    HEAD_ADDRESS,
    HEAD_HOST,
    WORKERS,
    WorkerHost,
)

RUN_HEAD_SCRIPT: Final[Path] = BASE_DIR / "run_head.sh"
RUN_WORKER_SCRIPT: Final[Path] = BASE_DIR / "run_worker.sh"
STOP_SCRIPT: Final[Path] = BASE_DIR / "stop_worker.sh"

REMOTE_HEAD_SH: Final[str] = "C:/Windows/Temp/run_ray_head.sh"
REMOTE_WORKER_SH: Final[str] = "C:/Windows/Temp/run_ray_worker.sh"
REMOTE_STOP_SH: Final[str] = "C:/Windows/Temp/stop_ray_worker.sh"
REMOTE_CODE_TAR: Final[str] = "C:/Windows/Temp/code.tar.gz"


def scp_file(host: WorkerHost, local_path: Path, remote_path: str) -> bool:
    """Copy a file to a remote Windows host via scp."""
    cmd = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        str(local_path),
        f"{host.user}@{host.ip}:{remote_path}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode == 0


async def run_supervisor(
    host: WorkerHost,
    script_local: Path,
    script_remote: str,
    stop_event: asyncio.Event,
    is_head: bool = False,
) -> None:
    """Manage a single remote node (head or worker) over SSH with automatic reconnection."""
    role = "head" if is_head else "worker"
    label = f"{host.name} ({role})"
    print(f"[{label}] Deploying script...")
    if not scp_file(host, script_local, script_remote):
        print(f"[{label}] ERROR: Failed to scp script to {host.ip}")
        return

    arg = "" if is_head else f" {HEAD_ADDRESS}"
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "BatchMode=yes",
        f"{host.user}@{host.ip}",
        f"wsl -d Ubuntu-24.04 -u root bash /mnt/c/{script_remote[3:]}{arg}",
    ]

    while not stop_event.is_set():
        print(f"[{label}] Launching process...")
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            async def stream_output() -> None:
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode().strip()
                    if text and not text.startswith("** WARNING:"):
                        print(f"[{label}] {text}")

            stream_task = asyncio.create_task(stream_output())
            wait_task = asyncio.create_task(proc.wait())
            stop_task = asyncio.create_task(stop_event.wait())

            done, pending = await asyncio.wait(
                [wait_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in pending:
                t.cancel()

            if stop_event.is_set():
                print(f"[{label}] Stopping process...")
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                await stream_task
                break

            await stream_task
            print(
                f"[{label}] SSH session ended (exit={proc.returncode}). Reconnecting in 5s..."
            )
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[{label}] Exception in supervisor: {e}")
            await asyncio.sleep(5)


async def cmd_up() -> None:
    """Bring up Ray cluster: start head node on pc01, then workers on pc02-pc06."""
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_signal() -> None:
        print("\nShutdown signal received. Stopping cluster...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    print(f"\n[cluster] Starting Ray Head on {HEAD_HOST.name} ({HEAD_HOST.ip})...")
    head_task = asyncio.create_task(
        run_supervisor(HEAD_HOST, RUN_HEAD_SCRIPT, REMOTE_HEAD_SH, stop_event, is_head=True)
    )

    # Allow head node 6 seconds to bind ports before starting workers
    await asyncio.sleep(6)

    print(f"[cluster] Launching {len(WORKERS)} worker nodes...")
    worker_tasks = [
        asyncio.create_task(
            run_supervisor(w, RUN_WORKER_SCRIPT, REMOTE_WORKER_SH, stop_event, is_head=False)
        )
        for w in WORKERS
    ]

    all_tasks = [head_task, *worker_tasks]
    print(f"\n[cluster] Supervising Ray cluster (1 head + {len(WORKERS)} workers). Press Ctrl+C to stop.\n")
    await asyncio.gather(*all_tasks, return_exceptions=True)
    print("[cluster] Cluster supervision ended.")


def cmd_down(stop_head_too: bool = True) -> None:
    """Stop Ray processes on remote workers and optionally the head."""
    hosts_to_stop = list(WORKERS)
    if stop_head_too:
        hosts_to_stop.append(HEAD_HOST)

    print(f"[cluster] Stopping Ray on {len(hosts_to_stop)} node(s)...")
    for host in hosts_to_stop:
        print(f"[{host.name}] Sending stop signal...")
        if scp_file(host, STOP_SCRIPT, REMOTE_STOP_SH):
            cmd = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                f"{host.user}@{host.ip}",
                f"wsl -d Ubuntu-24.04 -u root bash /mnt/c/{REMOTE_STOP_SH[3:]}",
            ]
            subprocess.run(cmd, capture_output=True, check=False)
        else:
            print(f"[{host.name}] Failed to reach host {host.ip}")

    print("[cluster] Down command completed.")


def cmd_status() -> None:
    """Print current Ray cluster status directly from head node."""
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        f"{HEAD_HOST.user}@{HEAD_HOST.ip}",
        "wsl -d Ubuntu-24.04 -u root /usr/local/bin/uv run --directory /root/rllib --no-sync ray status",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = [
        line
        for line in (res.stdout + res.stderr).splitlines()
        if not line.startswith("** WARNING:")
    ]
    print("\n".join(lines))


def cmd_sync() -> bool:
    """Package local src/ into a tarball and deploy to HEAD_HOST and all WORKERS in /root/rllib/."""
    tar_path = Path("/tmp/los_podadores_code.tar.gz")
    print(f"[sync] Packaging source code from {PROJECT_ROOT}/src...")
    src_dir = PROJECT_ROOT / "src"
    with tarfile.open(tar_path, "w:gz") as tar:
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__", ".git"}]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                file_path = Path(root) / f
                arcname = file_path.relative_to(PROJECT_ROOT)
                tar.add(file_path, arcname=arcname)

    all_hosts = [HEAD_HOST, *WORKERS]
    print(f"[sync] Uploading archive ({tar_path.stat().st_size / 1024:.1f} KB) to {len(all_hosts)} cluster nodes...")
    for host in all_hosts:
        if not scp_file(host, tar_path, REMOTE_CODE_TAR):
            print(f"[sync] Failed to upload code archive to {host.name}")
            return False

        unpack_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "BatchMode=yes",
            f"{host.user}@{host.ip}",
            f"wsl -d Ubuntu-24.04 -u root tar -xzf /mnt/c/{REMOTE_CODE_TAR[3:]} -C /root/rllib",
        ]
        res = subprocess.run(unpack_cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"[sync] Error unpacking code archive on {host.name}: {res.stderr}")
            return False
        print(f"[sync] Successfully synchronized code to {host.name}:/root/rllib/src")

    return True


def cmd_train(extra_args: list[str]) -> None:
    """Sync code and run distributed training on the cluster head."""
    if not cmd_sync():
        print("[train] Sync failed. Aborting training.")
        sys.exit(1)

    clean_vid_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        f"{HEAD_HOST.user}@{HEAD_HOST.ip}",
        f"wsl -d Ubuntu-24.04 -u root bash -c \"rm -rf /root/rllib/logs/videos/* /mnt/c/Users/{HEAD_HOST.user}/videos/*\"",
    ]
    subprocess.run(clean_vid_cmd, capture_output=True, check=False)

    args_str = " ".join(extra_args)
    train_cmd = (
        f"PYTHONUNBUFFERED=1 /root/rllib/.venv/bin/python /root/rllib/src/v2/train_car.py --cluster {args_str}"
    )
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        f"{HEAD_HOST.user}@{HEAD_HOST.ip}",
        f"wsl -d Ubuntu-24.04 -u root {train_cmd}",
    ]

    print(f"\n[train] Executing distributed training on {HEAD_HOST.name}:", flush=True)
    print(f"        {train_cmd}\n", flush=True)
    proc = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.strip()
        if stripped and not stripped.startswith("** WARNING:"):
            print(f"[remote] {stripped}", flush=True)
    proc.wait()

    # Pull down any generated evaluation videos
    print("\n[train] Syncing evaluation videos back to local logs/videos/...")
    local_vid_dir = PROJECT_ROOT / "logs" / "videos"
    local_vid_dir.mkdir(parents=True, exist_ok=True)
    stage_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        f"{HEAD_HOST.user}@{HEAD_HOST.ip}",
        f"wsl -d Ubuntu-24.04 -u root bash -c \"mkdir -p /mnt/c/Users/{HEAD_HOST.user}/videos && cp -f /root/rllib/logs/videos/*.mp4 /mnt/c/Users/{HEAD_HOST.user}/videos/ 2>/dev/null || true\"",
    ]
    subprocess.run(stage_cmd, capture_output=True, check=False)

    fetch_cmd = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        f"{HEAD_HOST.user}@{HEAD_HOST.ip}:C:/Users/{HEAD_HOST.user}/videos/*.mp4",
        str(local_vid_dir),
    ]
    fetch_res = subprocess.run(fetch_cmd, capture_output=True, text=True, check=False)
    if fetch_res.returncode == 0:
        vids = list(local_vid_dir.glob("*.mp4"))
        print(f"[train] Successfully synced {len(vids)} evaluation video(s) to {local_vid_dir}")
    else:
        print("[train] No new videos found or sync skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray RLlib cluster manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("up", help="Start cluster head (pc01) and workers (pc02-pc06)")

    down_p = subparsers.add_parser("down", help="Stop cluster workers and head")
    down_p.add_argument(
        "--workers-only",
        action="store_true",
        help="Leave head node running",
    )

    subparsers.add_parser("status", help="Show cluster status")
    subparsers.add_parser("sync", help="Sync latest code to cluster head")

    subparsers.add_parser("train", help="Run distributed training on cluster")
    subparsers.add_parser("eval", help="Evaluate trained policy in Gazebo")

    args, unknown = parser.parse_known_args()

    match args.command:
        case "up":
            asyncio.run(cmd_up())
        case "down":
            cmd_down(stop_head_too=not args.workers_only)
        case "status":
            cmd_status()
        case "sync":
            cmd_sync()
        case "train":
            cmd_train(unknown)
        case "eval":
            from eval_car import main as eval_main
            sys.argv = [sys.argv[0], *unknown]
            eval_main()
        case _:
            raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
