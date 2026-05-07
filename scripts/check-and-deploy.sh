#!/usr/bin/env bash
set -euo pipefail

cd /var/www/enlitenment

# Fetch latest remote state without merging
git fetch origin main

LOCAL_HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(git rev-parse origin/main)
DEPLOYED=$(cat _output/.deployed-commit 2>/dev/null || echo "")

# Skip only when local is up to date AND that commit is what is deployed.
# A failed build leaves DEPLOYED != LOCAL_HEAD, so the next tick retries.
if [ "$LOCAL_HEAD" = "$REMOTE_HEAD" ] && [ "$DEPLOYED" = "$LOCAL_HEAD" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Up to date (deployed: ${DEPLOYED:0:7}), skipping build"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') — Build needed (HEAD: ${LOCAL_HEAD:0:7}, remote: ${REMOTE_HEAD:0:7}, deployed: ${DEPLOYED:0:7}), running pull-build-deploy"
exec /var/www/enlitenment/scripts/pull-build-deploy.sh
