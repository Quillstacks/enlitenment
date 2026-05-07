#!/usr/bin/env bash
set -euo pipefail

cd /var/www/enlitenment
source .env

discord_notify() {
    local message="$1"
    curl -s -H "Content-Type: application/json" \
         -d "{\"content\": \"$message\"}" \
         "$DISCORD_WEBHOOK_URL" > /dev/null 2>&1 || true
}

# Trap errors — send failure message with the failing command
trap 'discord_notify "❌ **enlitenment build failed** at line $LINENO: \`$BASH_COMMAND\`\n$(date +%Y-%m-%d\ %H:%M:%S)"' ERR

echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting pull-build-deploy"

# Pull latest
git pull --ff-only origin main
COMMIT=$(git log -1 --format='%h %s')

# Build JupyterLite at root
source .venv/bin/activate
pip install -q -r requirements.txt

# Validate that all notebook imports are covered by requirements-pyodide.txt
python3 scripts/check-notebook-deps.py

# Build into a temp copy of content so we can inject %pip install cells
# without modifying the git-tracked source notebooks
rm -rf _output _build_content
cp -r content _build_content
python3 scripts/check-notebook-deps.py --inject _build_content
jupyter lite build --contents _build_content --output-dir _output
rm -rf _build_content

# Inject custom CSS into the built output
mkdir -p _output/custom
cp custom/custom.css _output/custom/custom.css

# Copy landing page
mkdir -p _output/landing
cp landing/index.html _output/landing/index.html
cp landing/favicon.svg _output/landing/favicon.svg

# Record the commit that was successfully built — used by check-and-deploy.sh
# to detect a stale or failed deploy and retry.
git rev-parse HEAD > _output/.deployed-commit

echo "$(date '+%Y-%m-%d %H:%M:%S') — Done"
discord_notify "✅ **enlitenment deployed** — \`$COMMIT\`\n$(date '+%Y-%m-%d %H:%M:%S')"
