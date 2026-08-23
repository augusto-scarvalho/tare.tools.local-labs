#!/usr/bin/env bash
set -euo pipefail

cli_link=/usr/local/bin/docker
cli_target=/mnt/wsl/docker-desktop/cli-tools/usr/bin/docker
agent_cli_link=/usr/bin/docker
socket_link=/run/docker.sock
socket_target=/mnt/wsl/docker-desktop/shared-sockets/guest-services/Ubuntu-24.04.docker.container-proxy.sock
config=/home/augus/.docker/config.json
config_sha=ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356

if [[ -L "$cli_link" && "$(readlink "$cli_link")" == "$cli_target" ]]; then
  unlink "$cli_link"
fi
if [[ -L "$agent_cli_link" && "$(readlink "$agent_cli_link")" == "$cli_target" ]]; then
  unlink "$agent_cli_link"
fi
if [[ -L "$socket_link" && "$(readlink "$socket_link")" == "$socket_target" ]]; then
  unlink "$socket_link"
fi
if [[ -S "$socket_link" && "$(stat -c '%g' "$socket_link")" == "989" ]]; then
  unlink "$socket_link"
fi
if [[ -f "$config" && "$(sha256sum "$config" | cut -d' ' -f1)" == "$config_sha" ]]; then
  unlink "$config"
fi
if getent group docker >/dev/null; then
  gpasswd -d augus docker >/dev/null 2>&1 || true
  groupdel docker
fi

test ! -e "$cli_link"
test ! -e "$agent_cli_link"
test ! -e "$socket_link"
test ! -e "$config"
! getent group docker >/dev/null
echo "temporary Docker WSL bridge removed"
