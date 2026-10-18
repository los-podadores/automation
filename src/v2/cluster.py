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
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import ray
from cluster_config import (
    HEAD_ADDRESS,
    HEAD_IP,
    HEAD_PORT,
    HEAD_TMPDIR,
    WORKERS,
    WorkerHost,
)

BASE_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = BASE_DIR / "run_worker.sh"
STOP_SCRIPT = BASE_DIR / "stop_worker.sh"
REMOTE_TEMP_SH = "C:/Windows/Temp/run_ray_worker.sh"
REMOTE_STOP_SH = "C:/Windows/Temp/stop_ray_worker.sh"


def is_head_running() -> bool:
    """Check if the local Ray head node is active and reachable."""
    env = os.environ.copy()
    env["TMPDIR"] = HEAD_TMPDIR
    env["RAY_AUTH_MODE"] = "disabled"
    result = subprocess.run(
        ["ray", "status", "--address", HEAD_ADDRESS],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return result.returncode == 0


def start_head() -> None:
    """Start local Ray head node if not already running."""
    if is_head_running():
        print(f"[head] Ray head already running at {HEAD_ADDRESS}")
        return

    print(f"[head] Starting Ray head at {HEAD_ADDRESS}...")
    env = os.environ.copy()
    env["TMPDIR"] = HEAD_TMPDIR
    env["RAY_AUTH_MODE"] = "disabled"
    Path(HEAD_TMPDIR).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            f"--node-ip-address={HEAD_IP}",
            f"--port={HEAD_PORT}",
            "--disable-usage-stats",
        ],
        check=True,
        env=env,
    )
    print(f"[head] Ray head successfully started at {HEAD_ADDRESS}")


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


async def run_worker_supervisor(
    host: WorkerHost, stop_event: asyncio.Event
) -> None:
    """Manage a single remote worker over SSH with automatic reconnection."""
    print(f"[{host.name}] Deploying run script...")
    if not scp_file(host, RUN_SCRIPT, REMOTE_TEMP_SH):
        print(f"[{host.name}] ERROR: Failed to scp run script to {host.ip}")
        return

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
        f"wsl -d Ubuntu-24.04 -u root bash /mnt/c/{REMOTE_TEMP_SH[3:]}",
    ]

    while not stop_event.is_set():
        print(f"[{host.name}] Launching worker process...")
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
                        print(f"[{host.name}] {text}")

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
                print(f"[{host.name}] Stopping worker process...")
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
                f"[{host.name}] SSH session ended (exit={proc.returncode}). Reconnecting in 5s..."
            )
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[{host.name}] Exception in supervisor: {e}")
            await asyncio.sleep(5)


async def cmd_up() -> None:
    """Bring up cluster head and all workers."""
    start_head()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_signal() -> None:
        print("\nShutdown signal received. Stopping workers...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    tasks = [
        asyncio.create_task(run_worker_supervisor(w, stop_event))
        for w in WORKERS
    ]

    print(f"\n[cluster] Supervising {len(WORKERS)} worker nodes. Press Ctrl+C to stop.\n")
    await asyncio.gather(*tasks, return_exceptions=True)
    print("[cluster] All workers stopped cleanly.")


def cmd_down(stop_head_too: bool = False) -> None:
    """Stop Ray workers on all remote hosts."""
    print("[cluster] Stopping all remote Ray workers...")
    for host in WORKERS:
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

    if stop_head_too:
        print("[head] Stopping local Ray head...")
        env = os.environ.copy()
        env["TMPDIR"] = HEAD_TMPDIR
        subprocess.run(["ray", "stop", "--force"], env=env, check=False)

    print("[cluster] Down command completed.")


def cmd_status() -> None:
    """Print current Ray cluster status."""
    env = os.environ.copy()
    env["TMPDIR"] = HEAD_TMPDIR
    env["RAY_AUTH_MODE"] = "disabled"
    subprocess.run(["ray", "status", "--address", HEAD_ADDRESS], env=env, check=False)


def cmd_test() -> None:
    """Run a distributed test confirming task execution across every node."""
    os.environ["TMPDIR"] = HEAD_TMPDIR
    os.environ["RAY_ENABLE_UV_RUNTIME_ENV"] = "0"
    os.environ["RAY_AUTH_MODE"] = "disabled"
    print(f"[test] Connecting to Ray cluster at {HEAD_ADDRESS}...")
    ray.init(
        address=HEAD_ADDRESS,
        ignore_reinit_error=True,
        runtime_env={"working_dir": str(BASE_DIR)},
    )

    nodes = ray.nodes()
    active_nodes = [n for n in nodes if n["Alive"]]
    print(f"[test] Active nodes in cluster: {len(active_nodes)} / {len(nodes)}")

    for n in active_nodes:
        ip = n["NodeManagerAddress"]
        hostname = n["NodeName"]
        cpus = n["Resources"].get("CPU", 0)
        gpus = n["Resources"].get("GPU", 0)
        print(f"  - Node {n['NodeID'][:8]} @ {ip} ({hostname}): {cpus} CPUs, {gpus} GPUs")

    @ray.remote
    def ping_node() -> dict[str, str | float]:
        import socket
        import time

        return {
            "hostname": socket.gethostname(),
            "time": time.time(),
        }

    print("\n[test] Dispatching test tasks to each active node...")
    futures = []
    for n in active_nodes:
        ip = n["NodeManagerAddress"]
        res_key = f"node:{ip}"
        f = ping_node.options(resources={res_key: 1}).remote()
        futures.append((ip, f))

    success = True
    for ip, fut in futures:
        try:
            res = ray.get(fut, timeout=30)
            print(f"  [OK] Node {ip} responded: {res}")
        except Exception as e:
            print(f"  [FAIL] Node {ip} error: {e}")
            success = False

    ray.shutdown()
    if not success:
        print("\n[test] Test FAILED on one or more nodes.")
        sys.exit(1)
    print("\n[test] All active nodes passed distributed test successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray RLlib cluster manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("up", help="Start cluster and workers")

    down_p = subparsers.add_parser("down", help="Stop cluster workers")
    down_p.add_argument(
        "--head",
        action="store_true",
        help="Also stop the local head node",
    )

    subparsers.add_parser("status", help="Show cluster status")
    subparsers.add_parser("test", help="Run distributed test on active nodes")

    args = parser.parse_args()

    match args.command:
        case "up":
            asyncio.run(cmd_up())
        case "down":
            cmd_down(stop_head_too=args.head)
        case "status":
            cmd_status()
        case "test":
            cmd_test()
        case _:
            raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
