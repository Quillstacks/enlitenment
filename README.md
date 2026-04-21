# enlitenment

Interactive, browser-based lecture notebooks for numerical methods and unsupervised deep learning. Everything runs client-side via [JupyterLite](https://jupyterlite.readthedocs.io/) and [Pyodide](https://pyodide.org/), so students only need a browser.

Two courses currently ship:

- **Numerical Methods** (`content/numerical-methods/`), lectures 01 to 06
- **Unsupervised Deep Learning** (`content/unsupervised-deep-learning/`), lectures 01 to 05

Each lecture directory holds a student notebook, a solutions notebook, a `checks.py` auto-grader, and any dataset files.

## Run locally

```bash
git clone https://github.com/Quillstacks/enlitenment.git
cd enlitenment
./build.sh
python3 -m http.server --directory _output 8000
```

Open `http://localhost:8000`. The first build downloads the Pyodide runtime and takes a minute; subsequent builds are fast.

## Work on a notebook without rebuilding

Open the `.ipynb` file directly in Jupyter Lab or VS Code. Install the kernel dependencies:

```bash
pip install -r requirements-notebooks.txt jupyterlab
```

Rebuild only when you want to test the deployed view (JupyterLite + Pyodide behave slightly differently from a local kernel, see `deployment.md`).

## Repository layout

| Path | Purpose |
|------|---------|
| `content/` | Source notebooks, datasets, and checkers (input to the build) |
| `landing/index.html` | Course landing page |
| `build.sh` | One-shot local build script |
| `deploy/` | Production nginx config |
| `scripts/` | Cron-based auto-deploy |
| `_output/` | Built static site (git-ignored) |

## Documentation

- [`deployment.md`](deployment.md), build pipeline, nginx config, capacity planning
- [`notebookskill.md`](notebookskill.md), notebook design conventions (the style bible for contributors)
- [`CONTRIBUTING.md`](CONTRIBUTING.md), how to propose a new lecture or fix an existing one

## License

Licensed under CC BY-NC-SA 4.0, see [`LICENSE`](LICENSE). You may share and adapt the material for non-commercial purposes as long as you credit the author and share derivatives under the same license. For commercial use, contact the author.
