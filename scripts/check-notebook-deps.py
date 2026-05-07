#!/usr/bin/env python3
"""Scan notebooks for third-party imports and verify they are covered by
requirements-pyodide.txt.  Exits non-zero if any import is missing.

Also used by the build to inject %pip install cells — see --inject flag.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Canonical Python stdlib list from the running interpreter (3.10+), plus
# IPython / Jupyter / pyodide names that are always available in the kernel.
STDLIB = set(sys.stdlib_module_names) | {
    "IPython", "ipywidgets", "piplite", "micropip", "pyodide", "pyodide_js",
    "js",
}

# Packages bundled with pyodide and always available without %pip install.
PYODIDE_BUILTINS = {
    "numpy", "matplotlib", "pandas", "scipy", "sympy", "networkx",
    "PIL", "Pillow", "cycler", "dateutil", "pyparsing", "pytz", "six",
    "packaging", "kiwisolver", "certifi", "charset_normalizer", "idna",
    "requests",
}

# Map import name → pip package name (only where they differ).
IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "dateutil": "python-dateutil",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "attr": "attrs",
    "gi": "pygobject",
}


def parse_pyodide_requirements(req_path: Path) -> set[str]:
    """Return the set of pip package names from requirements-pyodide.txt."""
    if not req_path.exists():
        return set()
    pkgs = set()
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # strip version specifiers
            pkgs.add(re.split(r"[><=!;]", line)[0].strip().lower())
    return pkgs


def extract_imports_from_source(source: str) -> set[str]:
    """Return top-level import names from Python source code."""
    imports = set()
    for line in source.split("\n"):
        line = line.strip()
        m = re.match(r"^(?:from|import)\s+([\w.]+)", line)
        if m:
            imports.add(m.group(1).split(".")[0])
    return imports


def extract_imports(nb_path: Path) -> set[str]:
    """Return top-level import names from a notebook's code cells."""
    with open(nb_path) as f:
        nb = json.load(f)
    imports = set()
    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        imports.update(extract_imports_from_source(source))
    return imports


def extract_imports_from_py(py_path: Path) -> set[str]:
    """Return top-level import names from a .py file."""
    return extract_imports_from_source(py_path.read_text())


def local_modules(content_dir: Path) -> dict[str, list[Path]]:
    """Return mapping of module name → list of .py file paths."""
    modules: dict[str, list[Path]] = {}
    for py in content_dir.rglob("*.py"):
        modules.setdefault(py.stem, []).append(py)
    return modules


def resolve_transitive_imports(
    direct_imports: set[str],
    local: dict[str, list[Path]],
    nb_path: Path,
) -> set[str]:
    """Expand imports by following local .py modules to find their third-party
    dependencies.  Returns the full set of third-party import names."""
    all_imports = set(direct_imports)
    visited: set[str] = set()
    queue = [m for m in direct_imports if m in local]

    while queue:
        mod = queue.pop()
        if mod in visited:
            continue
        visited.add(mod)
        # Find the .py file closest to the notebook (same directory first)
        nb_dir = nb_path.parent
        candidates = local[mod]
        best = None
        for c in candidates:
            if c.parent == nb_dir:
                best = c
                break
        if best is None:
            best = candidates[0]
        py_imports = extract_imports_from_py(best)
        all_imports.update(py_imports)
        # Follow any newly discovered local modules
        for imp in py_imports:
            if imp in local and imp not in visited:
                queue.append(imp)

    return all_imports


def inject_pip_install(nb_path: Path, packages: list[str]) -> bool:
    """Prepend a %pip install line to the first code cell of the notebook.
    Returns True if the notebook was modified."""
    with open(nb_path) as f:
        nb = json.load(f)

    pip_line = "%pip install -q " + " ".join(sorted(packages))

    # Check if there's already a %pip install anywhere
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "%pip install" in src:
                return False

    # Find the first code cell and prepend the install line into it
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            existing = cell["source"]
            if isinstance(existing, str):
                existing = [existing]
            cell["source"] = [pip_line + "\n"] + existing
            break

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inject",
        metavar="DIR",
        help="Inject %%pip install cells into notebooks under DIR",
    )
    parser.add_argument(
        "--content-dir",
        default="content",
        help="Directory containing source notebooks (default: content)",
    )
    parser.add_argument(
        "--requirements",
        default="requirements-pyodide.txt",
        help="Path to requirements-pyodide.txt",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    content_dir = root / args.content_dir
    req_path = root / args.requirements

    pyodide_pkgs = parse_pyodide_requirements(req_path)
    local = local_modules(content_dir)
    local_names = set(local.keys())

    # Build reverse map: package-name → set of import names it covers
    pkg_covers_import = {}
    for imp, pkg in IMPORT_TO_PACKAGE.items():
        pkg_covers_import.setdefault(pkg.lower(), set()).add(imp)
    # Also: any package whose name matches the import name
    for pkg in pyodide_pkgs:
        pkg_covers_import.setdefault(pkg, set()).add(pkg.replace("-", "_"))

    # Set of import names satisfied by requirements-pyodide.txt
    covered_imports = set()
    for pkg in pyodide_pkgs:
        covered_imports.update(pkg_covers_import.get(pkg, {pkg.replace("-", "_")}))

    errors = []
    inject_dir = Path(args.inject) if args.inject else None

    for nb_path in sorted(content_dir.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in str(nb_path):
            continue
        direct_imports = extract_imports(nb_path)
        # Follow local .py helpers to find transitive third-party imports
        all_imports = resolve_transitive_imports(direct_imports, local, nb_path)
        third_party = all_imports - STDLIB - PYODIDE_BUILTINS - local_names

        # Which of these are NOT covered by requirements-pyodide.txt?
        missing = third_party - covered_imports
        if missing:
            rel = nb_path.relative_to(root)
            errors.append((rel, missing))

        # Injection mode: add %pip install cell for packages this notebook needs
        if inject_dir and third_party:
            # Find which packages from requirements-pyodide.txt this notebook needs
            needed_pkgs = []
            for imp in sorted(third_party):
                pkg = IMPORT_TO_PACKAGE.get(imp, imp).lower()
                if pkg in pyodide_pkgs:
                    needed_pkgs.append(pkg)

            if needed_pkgs:
                # Compute the mirrored path under inject_dir
                rel = nb_path.relative_to(content_dir)
                target = inject_dir / rel
                if target.exists():
                    inject_pip_install(target, needed_pkgs)

    if errors:
        print("ERROR: The following notebooks import packages not listed in "
              "requirements-pyodide.txt:\n", file=sys.stderr)
        for rel, missing in errors:
            print(f"  {rel}:", file=sys.stderr)
            for m in sorted(missing):
                pkg = IMPORT_TO_PACKAGE.get(m, m)
                print(f"    import {m}  →  add '{pkg}' to requirements-pyodide.txt",
                      file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)

    print(f"OK — all notebook imports are covered "
          f"({len(pyodide_pkgs)} packages in requirements-pyodide.txt)")


if __name__ == "__main__":
    main()
