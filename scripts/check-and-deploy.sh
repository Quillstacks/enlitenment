#!/usr/bin/env bash
set -euo pipefail

cd /var/www/enlitenment

# Fetch latest remote state without merging
git fetch origin main

LOCAL_HEAD=$(git rev-parse HEAD)
BEHIND=$(git rev-list --count HEAD..origin/main)
DEPLOYED=$(cat _output/.deployed-commit 2>/dev/null || echo "")

# Skip only when remote has nothing new for us AND our HEAD is what is deployed.
# A failed build leaves DEPLOYED != LOCAL_HEAD, so the next tick retries.
# Local commits ahead of remote are fine — we don't rebuild for them on cron.
if [ "$BEHIND" -eq 0 ] && [ "$DEPLOYED" = "$LOCAL_HEAD" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Up to date (deployed: ${DEPLOYED:0:7}), skipping build"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') — Build needed (HEAD: ${LOCAL_HEAD:0:7}, behind: ${BEHIND}, deployed: ${DEPLOYED:0:7}), running pull-build-deploy"
exec /var/www/enlitenment/scripts/pull-build-deploy.sh
