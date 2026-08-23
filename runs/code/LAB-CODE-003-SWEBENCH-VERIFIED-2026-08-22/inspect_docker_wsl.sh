#!/usr/bin/env bash
set -u

echo PATHS
find /mnt/wsl -maxdepth 6 \( -name docker -o -name docker.sock \) -print 2>/dev/null | head -80
echo MOUNTS
mount | grep -E 'docker-desktop|/mnt/wsl' | head -80 || true
echo SOCKETS
find /mnt/wsl -type s -print 2>/dev/null | head -80
echo PROXY_TESTS
for proxy in \
  /mnt/wsl/docker-desktop/shared-sockets/guest-services/docker.proxy.sock \
  /mnt/wsl/docker-desktop/shared-sockets/host-services/docker.proxy.sock \
  /mnt/wsl/docker-desktop/shared-sockets/guest-services/Ubuntu-24.04.docker.container-proxy.sock
do
  echo "==== $proxy"
  DOCKER_HOST="unix://$proxy" timeout 8 /usr/local/bin/docker version 2>&1 | tail -12 || true
done
