"""Centralized configuration for the Ray RLlib cluster."""

from dataclasses import dataclass
from typing import Final

HEAD_IP: Final[str] = "100.64.0.21"
HEAD_PORT: Final[int] = 6379
HEAD_ADDRESS: Final[str] = f"{HEAD_IP}:{HEAD_PORT}"
HEAD_TMPDIR: Final[str] = "/tmp/ray-stable"
WORKER_TMPDIR: Final[str] = "/tmp/ray-worker"


@dataclass(frozen=True)
class WorkerHost:
    name: str
    ip: str
    user: str


HEAD_HOST: Final[WorkerHost] = WorkerHost(name="pc01", ip="100.64.0.21", user="UNSA04")

WORKERS: Final[tuple[WorkerHost, ...]] = (
    WorkerHost(name="pc02", ip="100.64.0.18", user="UNSA06"),
    WorkerHost(name="pc03", ip="100.64.0.10", user="UNSA09"),
    WorkerHost(name="pc04", ip="100.64.0.33", user="UNSA08"),
    WorkerHost(name="pc05", ip="100.64.0.35", user="UNSA"),
    WorkerHost(name="pc06", ip="100.64.0.14", user="UNSA10"),
)
