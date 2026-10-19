"""Provision Gazebo Harmonic and synchronize project code across all cluster workers."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "src" / "v2"
PROJECT_ROOT = BASE_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cluster_config import WORKERS, WorkerHost

INSTALL_SCRIPT = """#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

if command -v gz >/dev/null 2>&1; then
    echo "GZ_ALREADY_INSTALLED"
    exit 0
fi

curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list
apt update -qq
apt-get install -y --no-install-recommends gz-harmonic
echo "GZ_INSTALL_SUCCESS"
"""


async def provision_worker(worker: WorkerHost) -> bool:
    print(f"[{worker.name}] Starting provisioning...")
    local_script = Path(f"/tmp/install_gz_{worker.name}.sh")
    local_script.write_text(INSTALL_SCRIPT)

    # SCP install script
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        str(local_script),
        f"{worker.user}@{worker.ip}:C:/Windows/Temp/install_gz.sh",
    ]
    proc = await asyncio.create_subprocess_exec(*scp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if await proc.wait() != 0:
        print(f"[{worker.name}] Failed to upload install script.")
        return False

    # Execute install script in WSL
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{worker.user}@{worker.ip}",
        "wsl -d Ubuntu-24.04 -u root bash /mnt/c/Windows/Temp/install_gz.sh",
    ]
    proc = await asyncio.create_subprocess_exec(*ssh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode().strip()
        if "GZ_INSTALL_SUCCESS" in text or "GZ_ALREADY_INSTALLED" in text:
            print(f"[{worker.name}] {text}")
    ret = await proc.wait()
    if ret != 0:
        print(f"[{worker.name}] Gazebo installation exited with code {ret}")
        return False

    # SCP code tarball
    tar_path = Path("/tmp/los_podadores_code.tar.gz")
    print(f"[{worker.name}] Uploading code archive...")
    scp_tar = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        str(tar_path),
        f"{worker.user}@{worker.ip}:C:/Windows/Temp/code.tar.gz",
    ]
    proc = await asyncio.create_subprocess_exec(*scp_tar, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if await proc.wait() != 0:
        print(f"[{worker.name}] Failed to upload code archive.")
        return False

    # Unpack code in /root/rllib
    unpack_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{worker.user}@{worker.ip}",
        "wsl -d Ubuntu-24.04 -u root tar -xzf /mnt/c/Windows/Temp/code.tar.gz -C /root/rllib",
    ]
    proc = await asyncio.create_subprocess_exec(*unpack_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if await proc.wait() != 0:
        print(f"[{worker.name}] Failed to unpack code.")
        return False

    # Verify gzdrl import
    verify_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{worker.user}@{worker.ip}",
        "wsl -d Ubuntu-24.04 -u root bash -c 'export PYTHONPATH=/root/rllib:/root/rllib/src/sim/gzdrl/_lib; export LD_LIBRARY_PATH=/root/rllib/src/sim/gzdrl/_lib:$LD_LIBRARY_PATH; /usr/local/bin/uv run --directory /root/rllib --no-sync python -c \"import gzdrl; from gzdrl._core import DRLServer; print(DRLServer)\"'",
    ]
    proc = await asyncio.create_subprocess_exec(*verify_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    if b"DRLServer" in out:
        print(f"[{worker.name}] SUCCESS: gzdrl verified!")
        return True
    else:
        print(f"[{worker.name}] VERIFICATION FAILED: {out.decode()}")
        return False


async def main() -> None:
    results = await asyncio.gather(*(provision_worker(w) for w in WORKERS))
    if all(results):
        print("\nAll worker nodes successfully provisioned and verified with Gazebo Harmonic & GzDRL!")
    else:
        print("\nSome workers failed provisioning.")


if __name__ == "__main__":
    asyncio.run(main())
