#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt
jupyter lite build --contents content --output-dir _output

# Copy landing page
mkdir -p _output/landing
cp landing/index.html _output/landing/index.html
cp landing/favicon.svg _output/landing/favicon.svg
