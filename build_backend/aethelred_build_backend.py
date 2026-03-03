"""
Minimal vendored PEP 517 backend for the Aethelred Python SDK.

Purpose:
- Allow `python -m build` to succeed in isolated/no-network environments.
- Avoid runtime dependency on external build backends (e.g. hatchling) on CI or
  air-gapped developer machines.

This backend builds a pure-Python wheel and sdist from `pyproject.toml` (PEP 621).
It is intentionally small and only implements the hooks required by our workflows.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _load_pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _project() -> dict[str, Any]:
    return _load_pyproject()["project"]


def _norm_dist_name(name: str) -> str:
    # Wheel filenames normalize '-' to '_'
    return name.replace("-", "_")


def _sdist_base_name(name: str, version: str) -> str:
    return f"{_norm_dist_name(name)}-{version}"


def _dist_info_dir(name: str, version: str) -> str:
    return f"{_norm_dist_name(name)}-{version}.dist-info"


def _iter_package_files() -> Iterable[Path]:
    data = _load_pyproject()
    packages = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", ["aethelred"])
    )
    for pkg in packages:
        pkg_path = ROOT / pkg
        if not pkg_path.exists():
            continue
        for p in pkg_path.rglob("*"):
            if p.is_dir():
                continue
            if "__pycache__" in p.parts:
                continue
            if p.suffix in {".pyc", ".pyo"}:
                continue
            yield p


def _iter_sdist_files() -> Iterable[Path]:
    exclude_dirs = {"dist", "build", ".pytest_cache", ".mypy_cache", "__pycache__"}
    exclude_suffixes = {".pyc", ".pyo"}
    for p in ROOT.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(ROOT)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        if p.suffix in exclude_suffixes:
            continue
        yield p


def _format_requires_dist(dep: str, extra: str | None = None) -> str:
    dep = dep.strip()
    if extra is None:
        return dep
    if ";" in dep:
        base, marker = dep.split(";", 1)
        return f"{base.strip()} ; ({marker.strip()}) and extra == '{extra}'"
    return f"{dep} ; extra == '{extra}'"


def _metadata_text() -> str:
    p = _project()
    lines: list[str] = []
    lines.append("Metadata-Version: 2.1")
    lines.append(f"Name: {p['name']}")
    lines.append(f"Version: {p['version']}")
    if p.get("description"):
        lines.append(f"Summary: {p['description']}")
    if p.get("requires-python"):
        lines.append(f"Requires-Python: {p['requires-python']}")

    for author in p.get("authors", []):
        name = author.get("name")
        email = author.get("email")
        if name and email:
            lines.append(f"Author-email: {name} <{email}>")
        elif email:
            lines.append(f"Author-email: {email}")

    license_obj = p.get("license")
    if isinstance(license_obj, dict):
        if license_obj.get("text"):
            lines.append(f"License: {license_obj['text']}")

    keywords = p.get("keywords")
    if keywords:
        lines.append(f"Keywords: {','.join(keywords)}")

    for classifier in p.get("classifiers", []):
        lines.append(f"Classifier: {classifier}")

    for label, url in p.get("urls", {}).items():
        lines.append(f"Project-URL: {label}, {url}")

    for dep in p.get("dependencies", []):
        lines.append(f"Requires-Dist: {_format_requires_dist(dep)}")

    for extra, deps in p.get("optional-dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra}")
        for dep in deps:
            lines.append(f"Requires-Dist: {_format_requires_dist(dep, extra)}")

    readme = ROOT / p.get("readme", "README.md")
    body = ""
    if readme.exists():
        try:
            body = readme.read_text(encoding="utf-8")
        except Exception:
            body = ""
    return "\n".join(lines) + "\n\n" + body


def _wheel_text() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: aethelred_build_backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _entry_points_text() -> str:
    scripts = _project().get("scripts", {})
    if not scripts:
        return ""
    out = ["[console_scripts]"]
    for name, target in scripts.items():
        out.append(f"{name} = {target}")
    out.append("")
    return "\n".join(out)


def _hash_bytes(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _hash_file(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return _hash_bytes(data), len(data)


def _build_metadata_dir(target_dir: Path) -> str:
    p = _project()
    dist_info = target_dir / _dist_info_dir(p["name"], p["version"])
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    entry_points = _entry_points_text()
    if entry_points:
        (dist_info / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    return dist_info.name


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
) -> str:
    return _build_metadata_dir(Path(metadata_directory))


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    p = _project()
    name = p["name"]
    version = p["version"]
    wheel_name = f"{_norm_dist_name(name)}-{version}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        # package files
        for src in _iter_package_files():
            rel = src.relative_to(ROOT)
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

        dist_info_name = _build_metadata_dir(staging)
        dist_info_path = staging / dist_info_name

        # RECORD (written after all files exist)
        record_rows: list[tuple[str, str, str]] = []
        for f in sorted(staging.rglob("*")):
            if f.is_dir():
                continue
            rel = f.relative_to(staging).as_posix()
            h, size = _hash_file(f)
            record_rows.append((rel, h, str(size)))

        record_rel = f"{dist_info_name}/RECORD"
        record_file = dist_info_path / "RECORD"
        with record_file.open("w", encoding="utf-8", newline="") as rf:
            writer = csv.writer(rf)
            for row in record_rows:
                writer.writerow(row)
            writer.writerow((record_rel, "", ""))

        with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(staging.rglob("*")):
                if f.is_dir():
                    continue
                zf.write(f, f.relative_to(staging).as_posix())

    return wheel_name


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    p = _project()
    base = _sdist_base_name(p["name"], p["version"])
    filename = f"{base}.tar.gz"
    out_path = Path(sdist_directory) / filename

    with tarfile.open(out_path, "w:gz") as tf:
        for src in sorted(_iter_sdist_files()):
            rel = src.relative_to(ROOT)
            arcname = f"{base}/{rel.as_posix()}"
            tf.add(src, arcname=arcname, recursive=False)

    return filename
