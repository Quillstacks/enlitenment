#!/usr/bin/env bash
set -euo pipefail

cd /var/www/enlitenment

# Fetch latest remote state without merging
git fetch origin main

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — No changes, skipping build"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') — New commits detected, running pull-build-deploy"
exec /var/www/enlitenment/scripts/pull-build-deploy.sh
