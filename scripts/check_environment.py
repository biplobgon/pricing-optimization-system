"""Check whether the local development environment is ready."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "statsmodels",
    "fastapi",
    "uvicorn",
    "pydantic",
    "matplotlib",
    "seaborn",
    "plotly",
    "pytest",
]

REQUIRED_PATHS = [
    Path("README.md"),
    Path("requirements/base.txt"),
    Path("requirements/dev.txt"),
    Path("data/raw"),
    Path("src"),
    Path("tests"),
]


def check_python_version() -> bool:
    """Validate the active Python version."""
    version = sys.version_info
    ok = version >= (3, 9)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python {platform.python_version()} detected")
    return ok


def check_paths() -> bool:
    """Validate required project paths."""
    ok = True
    for path in REQUIRED_PATHS:
        exists = path.exists()
        status = "OK" if exists else "FAIL"
        print(f"[{status}] {path}")
        ok = ok and exists
    return ok


def check_packages() -> bool:
    """Validate required importable packages."""
    ok = True
    for package in REQUIRED_PACKAGES:
        exists = importlib.util.find_spec(package) is not None
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {package}")
        ok = ok and exists
    return ok


def check_dataset() -> bool:
    """Report whether the raw dataset is available."""
    dataset = Path("data/raw/retail_price.csv")
    exists = dataset.exists()
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {dataset}")
    return exists


def main() -> int:
    """Run all setup checks."""
    print("Pricing Optimization System environment check")
    print("=" * 52)

    python_ok = check_python_version()
    print()
    paths_ok = check_paths()
    print()
    packages_ok = check_packages()
    print()
    dataset_ok = check_dataset()

    print()
    if python_ok and paths_ok and packages_ok and dataset_ok:
        print("Environment is ready.")
        return 0

    print("Environment is not fully ready. Review the MISSING/FAIL lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
