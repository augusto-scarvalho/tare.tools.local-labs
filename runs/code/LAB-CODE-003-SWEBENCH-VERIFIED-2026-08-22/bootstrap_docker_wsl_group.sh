#!/usr/bin/env bash
set -euo pipefail

if ! getent group docker >/dev/null; then
  groupadd --system docker
fi
usermod -aG docker augus

cli=/mnt/wsl/docker-desktop/cli-tools/usr/bin/docker
proxy=/mnt/wsl/docker-desktop/shared-sockets/guest-services/Ubuntu-24.04.docker.container-proxy.sock
test -x "$cli"
test -S "$proxy"
chgrp docker "$proxy"
chmod 0660 "$proxy"
ln -sfn "$cli" /usr/local/bin/docker
ln -sfn "$proxy" /run/docker.sock

getent group docker
id augus
ls -l /usr/local/bin/docker /run/docker.sock "$proxy"
