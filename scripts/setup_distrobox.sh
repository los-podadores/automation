#!/usr/bin/env bash
# Idempotent provisioner for Gazebo Harmonic Distrobox container
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="gazebo-harmonic"

echo "=== [Distrobox Setup: $CONTAINER_NAME] ==="

if ! command -v distrobox >/dev/null 2>&1; then
    echo "Error: 'distrobox' is not installed or not in PATH." >&2
    echo "Enter via 'nix develop' or install distrobox to continue." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1 && ! command -v podman >/dev/null 2>&1; then
    echo "Error: Neither docker nor podman container runtime was found." >&2
    exit 1
fi

# Check if container already exists
if distrobox list | grep -qw "$CONTAINER_NAME"; then
    echo "Container '$CONTAINER_NAME' already exists."
    # Ensure running
    if [ "$(docker inspect --type container --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)" != "running" ]; then
        echo "Starting container '$CONTAINER_NAME'..."
        docker start "$CONTAINER_NAME" >/dev/null
    fi
else
    echo "Creating container '$CONTAINER_NAME' from $SCRIPT_DIR/distrobox.ini..."
    distrobox assemble create --file "$SCRIPT_DIR/distrobox.ini"
fi

# Verify Gazebo inside container
echo "Verifying Gazebo inside '$CONTAINER_NAME'..."
env -u LD_LIBRARY_PATH distrobox enter "$CONTAINER_NAME" -- bash -c 'export QT_QPA_PLATFORM=xcb; /usr/bin/gz sim --version'

echo "=== Distrobox setup completed successfully. ==="
