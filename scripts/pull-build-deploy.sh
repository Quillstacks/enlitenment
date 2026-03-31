#!/usr/bin/env bash
set -euo pipefail

cd /var/www/enlitenment

echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting pull-build-deploy"

# Pull latest
git pull --ff-only origin main

# Build
source .venv/bin/activate
pip install -q -r requirements.txt
jupyter lite build --contents content --output-dir _output

echo "$(date '+%Y-%m-%d %H:%M:%S') — Done"
