#!/usr/bin/env bash
set -euo pipefail

cd /var/www/enlitenment

echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting pull-build-deploy"

# Pull latest
git pull --ff-only origin main

# Build JupyterLite at root
source .venv/bin/activate
pip install -q -r requirements.txt
rm -rf _output
jupyter lite build --contents content --output-dir _output

# Inject custom CSS into the built output
mkdir -p _output/custom
cp custom/custom.css _output/custom/custom.css

# Copy landing page
mkdir -p _output/landing
cp landing/index.html _output/landing/index.html
cp landing/favicon.svg _output/landing/favicon.svg

echo "$(date '+%Y-%m-%d %H:%M:%S') — Done"
