#!/usr/bin/env bash
# Instalación explícita en Ubuntu 24.04. No modifica Windows ni instala ROS.
set -euo pipefail
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]]; then
  echo "Este instalador requiere Ubuntu 24.04; detectado: ${PRETTY_NAME:-desconocido}." >&2
  echo "En PowerShell: wsl --install -d Ubuntu-24.04" >&2
  echo "Conserva tu distribución actual y abre después Ubuntu-24.04." >&2
  exit 1
fi
if [[ "$(dpkg --print-architecture)" != amd64 ]]; then
  echo "Este entorno inicial requiere Ubuntu 24.04 amd64." >&2
  exit 1
fi
sudo -v
sudo apt-get update
sudo apt-get install -y curl gnupg lsb-release ca-certificates
KEY_FILE="$(mktemp)"
trap 'rm -f -- "$KEY_FILE"' EXIT
curl --fail --location --proto '=https' --tlsv1.2 \
  https://packages.osrfoundation.org/gazebo.gpg --output "$KEY_FILE"
sudo install -m 0644 "$KEY_FILE" /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable noble main' \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install -y gz-harmonic python3-gz-transport13 python3-gz-msgs10 python3-tk python3-numpy
/usr/bin/python3 -c 'from gz.transport13 import Node; from gz.msgs10.twist_pb2 import Twist; import tkinter; print("Dependencias del tractor instaladas")'
gz sim --versions
