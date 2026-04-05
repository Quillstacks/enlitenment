# Deployment Guide

This document describes how to build and serve the enlitenment JupyterLite site on a Linux server with nginx.

---

## Architecture

```
Student's browser                        Server (4-worker CPU)
┌──────────────────────┐                ┌──────────────────────┐
│  Landing page (HTML) │◄──── static ───│  nginx               │
│  JupyterLite (JS)    │     files      │    ├─ _output/       │
│  Pyodide (WASM)      │                │    │   index.html     │  ← landing page
│  Python kernel       │                │    │   notebooks/     │  ← JupyterLite
│  numpy, matplotlib   │                │    │     *.ipynb      │
│                      │                │    │     *.wasm       │
│  ALL execution is    │                │                      │
│  client-side         │                │  NO server-side      │
└──────────────────────┘                │  computation         │
                                        └──────────────────────┘
```

JupyterLite runs entirely in the browser. The server only serves static files. There are no kernels, no user sessions, and no server-side state. This means:

- **Scaling** is trivial — nginx with 4 workers handles thousands of concurrent static requests.
- **Security surface** is minimal — no code execution on the server.
- **Student data** lives only in the browser's localStorage. Clearing cache = losing work.

---

## Prerequisites

- Linux server with nginx installed
- Python 3.10+ with pip
- Git

---

## Build

### Manual build

```bash
cd /var/www/enlitenment

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install build dependencies
pip install -r requirements.txt

# Build JupyterLite into _output/notebooks/
jupyter lite build --contents content --output-dir _output/notebooks

# Copy the landing page to the output root
cp landing/index.html _output/index.html
```

The built site is entirely contained in `_output/`. Serve that directory.

### Automated build via cron

The script `scripts/pull-build-deploy.sh` pulls the latest code and rebuilds:

```bash
# Add to crontab (e.g., rebuild every 15 minutes)
*/15 * * * * /var/www/enlitenment/scripts/pull-build-deploy.sh >> /var/log/enlitenment-deploy.log 2>&1
```

This is useful for picking up new notebooks pushed to main without manual SSH.

---

## Nginx Configuration

A ready-to-use config is at `deploy/nginx-site.conf`. Copy it into nginx's sites directory:

```bash
sudo cp deploy/nginx-site.conf /etc/nginx/sites-available/enlitenment
sudo ln -s /etc/nginx/sites-available/enlitenment /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### What the config does

#### 1. CORS headers (required)

```nginx
add_header Cross-Origin-Opener-Policy  "same-origin" always;
add_header Cross-Origin-Embedder-Policy "require-corp" always;
```

Pyodide needs `SharedArrayBuffer`, which browsers only enable when these two headers are present. **Without them, JupyterLite will fail silently or fall back to single-threaded mode on Chrome/Edge.**

#### 2. Gzip compression

```nginx
gzip on;
gzip_types application/javascript application/wasm application/json text/css text/html;
```

Compresses text-based assets on the fly. The Pyodide WASM bundle is ~30 MB uncompressed but ~12 MB gzipped. This cuts initial load time significantly, especially when 50 students hit the page at once.

#### 3. Cache headers

| File type | Cache duration | Reason |
|-----------|---------------|--------|
| `.wasm`, `.js`, `.css` | 6 months, immutable | These don't change between builds. After one student loads the page, their browser caches everything. Repeat visits are instant. |
| `.ipynb`, `.html` | 10 minutes, must-revalidate | Notebooks may be updated between lectures. Short cache ensures students get fresh content without manual cache-clearing. |

#### 4. Connection limits

```nginx
limit_conn perip 20;
```

Prevents any single client from opening excessive connections. With 50 students this is a non-issue, but it's good hygiene.

### Adapting the config

- **Domain name:** Replace `server_name _;` with your actual domain.
- **HTTPS:** Add an `ssl` block or use certbot (`sudo certbot --nginx -d yourdomain.edu`).
- **Port:** Change `listen 80;` if needed.
- **Document root:** Adjust `root /var/www/enlitenment/_output;` to match your deployment path.

---

## Capacity Planning

### 50 students — will the server cope?

**Yes, easily.** The math:

- Initial page load: each student downloads ~12 MB (gzipped) of WASM + JS + notebooks.
- With 50 students arriving within a 2-minute window: ~600 MB total transfer.
- A 100 Mbps link can push 600 MB in ~50 seconds. A gigabit link handles it in 5 seconds.
- nginx with 4 workers can serve ~10,000 concurrent static file requests. 50 is negligible.

After the initial load, the server is essentially idle — all computation happens in the browser.

### What could go wrong

| Symptom | Cause | Fix |
|---------|-------|-----|
| Notebooks won't load, blank page | Missing COOP/COEP headers | Check nginx config has the two `add_header` lines |
| Extremely slow first load | No gzip, students downloading 30 MB each | Enable gzip in nginx |
| Students lose work after closing browser | localStorage cleared | Advise students to download notebooks. Consider adding a reminder banner. |
| `SharedArrayBuffer is not defined` in console | Headers not applied to `.js` files | Make sure `add_header` is repeated inside `location` blocks (nginx quirk: location-level headers override server-level) |
| Safari shows errors | Safari < 15 has incomplete WASM support | Advise Chrome or Firefox |
| Mobile devices crash | Pyodide needs ~200 MB RAM | Advise laptop/desktop only |

---

## Directory Structure

```
enlitenment/
├── build.sh                        # Local build script
├── deployment.md                   # This file
├── jupyter_lite_config.json        # JupyterLite build config
├── requirements.txt                # Python build dependencies
├── notebookskill.md                # Notebook design guide (not deployed)
│
├── landing/
│   └── index.html                  # Welcome page (copied to _output/)
│
├── deploy/
│   └── nginx-site.conf             # Production nginx config
│
├── scripts/
│   └── pull-build-deploy.sh        # Cron-based auto-deploy
│
├── content/                        # Source notebooks (input to build)
│   ├── .jupyterliteignore          # Excludes *_solutions.ipynb from build
│   ├── numerical-methods/
│   │   ├── lecture-01/             # 01_grid_search.ipynb + solutions + checks.py
│   │   ├── lecture-02/             # 02_floating_point.ipynb + ...
│   │   ├── lecture-03/             # 03_error_analysis.ipynb + ...
│   │   ├── lecture-04/             # 04_root_finding.ipynb + ...
│   │   ├── lecture-05/             # 05_global_optimization.ipynb + ...
│   │   ├── lecture-06/             # 06_integration.ipynb + ...
│   │   └── lecture-07/             # Raw .py files (not yet converted)
│   └── unsupervised-deep-learning/
│       └── lecture-01/             # 01_foundations.ipynb + solutions + checks.py
│
└── _output/                        # Built site (git-ignored, served by nginx)
    ├── index.html                  # Landing page
    └── notebooks/                  # JupyterLite app + all notebooks
```

---

## Adding New Notebooks

1. Create a new `lecture-NN/` directory under the appropriate course in `content/`.
2. Add the student notebook (`NN_topic.ipynb`), solution (`NN_topic_solutions.ipynb`), and `checks.py`.
3. Solution notebooks are automatically excluded from the build by `.jupyterliteignore`.
4. Push to main. If the cron job is active, the site rebuilds automatically.

---

## Browser Compatibility

| Browser | Status |
|---------|--------|
| Chrome / Edge (desktop) | Fully supported, best performance |
| Firefox (desktop) | Fully supported |
| Safari 15+ (macOS) | Supported, slightly slower |
| Safari (iOS) | Works but constrained by device RAM |
| Mobile Chrome/Firefox | Technically works, not recommended (RAM) |
| IE11 | Not supported (no WebAssembly) |

The landing page advises students to use Chrome or Firefox on a laptop.
