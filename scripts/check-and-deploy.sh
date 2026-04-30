#!/usr/bin/env bash
set -euo pipefail

cd /var/www/enlitenment

# Fetch latest remote state without merging
git fetch origin main

# Only rebuild when remote has commits that local doesn't
BEHIND=$(git rev-list --count HEAD..origin/main)

if [ "$BEHIND" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — No changes, skipping build"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') — New commits detected, running pull-build-deploy"
exec /var/www/enlitenment/scripts/pull-build-deploy.sh
