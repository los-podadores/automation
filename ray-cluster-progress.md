# Ray RLlib cluster — progress, architecture, and usage

Date: 2026-10-16. Head node: this laptop (`cricro-laptop`, tailscale `100.64.0.13`).
Goal: minimal working Ray+RLlib setup in `src/v2/`, venvs on all 6 lab PCs (WSL),
workers joined to this head, distributed RLlib smoke test green.

## Status: 11/11 — ALL TASKS COMPLETE ✅

| # | Task | State |
|---|------|-------|
| 1 | Minimal `src/v2/` Ray RLlib setup, local smoke green | ✅ Done |
| 2 | SSH probe all pc01–pc06 | ✅ Done (all answer) |
| 3 | WSL + Ubuntu on all hosts | ✅ Done (6/6 `Ubuntu-24.04`) |
| 4 | Minimal rsync (`pyproject`+`uv.lock` only) + venv install per host | ✅ Done |
| 5 | `ray[rllib]` verified in all 6 remote venvs (`ray 2.59.0 rllib OK`) | ✅ Done |
| 6 | Ray head on this host (`100.64.0.13:6379`) | ✅ Done, healthy |
| 7 | Headscale ACL fix so workers can reach head | ✅ Done (`mecanica@` group routing fixed) |
| 8 | Workers join head and **stay** joined | ✅ Done (resolved via supervised worker sessions) |
| 9 | Distributed task verification across all 7 nodes | ✅ Done (100% pass across 7 nodes) |
| 10 | Distributed RLlib PPO smoke test with 6 env runners | ✅ Done (`DISTRIBUTED RLLIB SMOKE OK`) |
| 11 | Cleanup scratch files and unified cluster manager CLI | ✅ Done (`src/v2/cluster.py`) |

---

## Cluster Overview

- **Nodes active**: 7 total (1 head laptop + 6 remote lab machines)
- **Total Compute**: 112.0 CPUs, 6.0 GPUs, 36.85 GiB RAM, 15.79 GiB object store
- **Networking**: Tailscale mesh over Headscale (`100.64.0.0/10`)

### Node Inventory

| Host | Tailscale IP | SSH user | Windows user | WSL Distro | Cores / GPU | Status |
|------|--------------|----------|--------------|------------|-------------|--------|
| **head** | 100.64.0.13 | cricro | cricro | Arch Linux | 16 CPU | Active |
| **pc01** | 100.64.0.21 | UNSA04 | unsa04 | Ubuntu-24.04, mirrored | 16 CPU, 1 GPU | Active |
| **pc02** | 100.64.0.18 | UNSA06 | unsa06 | Ubuntu-24.04, mirrored | 16 CPU, 1 GPU | Active |
| **pc03** | 100.64.0.10 | UNSA09 | unsa09 | Ubuntu-24.04, mirrored | 16 CPU, 1 GPU | Active |
| **pc04** | 100.64.0.33 | UNSA08 | unsa08 | Ubuntu-24.04, mirrored | 16 CPU, 1 GPU | Active |
| **pc05** | 100.64.0.35 | UNSA | unsa | Ubuntu-24.04, mirrored | 16 CPU, 1 GPU | Active |
| **pc06** | 100.64.0.14 | UNSA10 | unsa10 | Ubuntu-24.04, mirrored | 16 CPU, 1 GPU | Active |

---

## Root Cause & Technical Resolutions

### 1. The Worker Drop Blocker (WSL Lifecycle)
- **Root Cause**: On Windows 11 WSL2, the lightweight utility VM terminates whenever there are no active Windows processes (`wsl.exe` or GUI sessions) attached to the distribution. When workers ran `ray start` detached via `nohup` or `svc-start.sh` and the SSH session closed, the `wsl.exe` process exited. Windows detected 0 active clients and shut down the WSL VM ~30–60 seconds later, wiping `/tmp` and killing the raylet.
- **Why Task Scheduler was infeasible**: Scheduled Tasks created without stored user credentials run in `Interactive Only` mode (failing when no user is logged into the physical Windows console), and WSL 2 explicitly rejects running under `NT AUTHORITY\SYSTEM` (`WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED`).
- **Solution**: The head node acts as the cluster supervisor via `src/v2/cluster.py`. It holds managed, persistent SSH connections executing `ray start --address=100.64.0.13:6379 --block`. This keeps `wsl.exe` actively running on Windows, maintains the raylet throughout the experiment, and features automatic reconnection if transient network hiccups occur. On shutdown (Ctrl+C or `cluster.py down`), all resources on the lab PCs are cleanly released.

### 2. Raylet Worker Sub-process PATH (`uv: not found`)
- **Root Cause**: When raylet spawns worker task processes, it uses a non-login, non-interactive subshell. The default `PATH` in Ubuntu does not include `/root/.local/bin/`, causing tasks to fail with `bash: line 1: exec: uv: not found`.
- **Solution**: Symlinked `/root/.local/bin/uv` to `/usr/local/bin/uv` on all 6 worker nodes.

### 3. Working Directory Package Overhead
- **Root Cause**: Ray's default runtime environment zipped the entire repository (including model checkpoints like `best_model.zip` and `yolov8s.pt` totaling ~40 MB). Pushing 40 MB to 6 workers concurrently over Tailscale caused task startup timeouts (>15–30 s) on some machines.
- **Solution**: Scoped `runtime_env={"working_dir": str(BASE_DIR)}` to `src/v2/` (~100 KB) and set `RAY_ENABLE_UV_RUNTIME_ENV=0` to prevent `uv` project directory path mismatch warnings between host and container paths.

---

## Cluster Management CLI (`src/v2/cluster.py`)

A unified orchestrator was implemented in `src/v2/cluster.py` with the following entry points:

```bash
# 1. Start the cluster head and all 6 remote workers (with auto-reconnect)
uv run python src/v2/cluster.py up

# 2. Check cluster status and active node resources (112 CPU, 4 GPU)
uv run python src/v2/cluster.py status

# 3. Run distributed ping verification across all nodes
uv run python src/v2/cluster.py test

# 4. Stop all remote workers cleanly
uv run python src/v2/cluster.py down
# Or stop workers AND the local head:
uv run python src/v2/cluster.py down --head
```

---

## Distributed RLlib Smoke Test

The distributed test runs PPO on `CartPole-v1` with 6 distributed `EnvRunners` executing across the 6 worker machines:

```bash
PYTHONPATH=. VIRTUAL_ENV=$PWD/src/v2/.venv TMPDIR=/tmp/ray-stable \
  uv run --active --project src/v2 python src/v2/smoke_rllib.py
```

Result:
```text
Connecting to Ray cluster at 100.64.0.13:6379...
Connected to cluster with 7 active nodes.
Configuring PPO with 6 distributed env runners...
iter=0 episode_return_mean=30.50
iter=1 episode_return_mean=32.15
iter=2 episode_return_mean=38.76
DISTRIBUTED RLLIB SMOKE OK
```
